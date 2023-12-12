import pickle
import numpy as np
import torch
import tqdm

from common.ftl_base import FTLBase
from common.ftl_param import FTLParam
from utils.ftl_log import LOGGER
from phe import paillier
from utils import config
from utils.ftl_data_loader import FTLDataLoader
from utils.timer import timer


class FTLHost(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLHost, self).__init__(param)
        self.nab_ids = None
        self.nab_indices = []  # nab indices refer to data_matrix
        self.nc_ids = None
        self.nc_indices = []  # nc indices refer to nab

        self.__get_overlap_id()
        # generate key pairx
        self._public_key, self._private_key = paillier.generate_paillier_keypair()
        LOGGER.info("ftl host initialization finished")

    def __get_overlap_id(self):
        """
        get nab, nc and the corresponding indices to these ids
        """
        self.send(
            pickle.dumps(list(self.data_loader.id_index_map.keys()))
        )  # send ids to guest

        self.nab_ids = pickle.loads(self.rcv())  # rcv nab ids
        LOGGER.debug(f"host get nab ids: {self.nab_ids}")
        for id in self.nab_ids:
            self.nab_indices.append(self.data_loader.id_index_map[id])

        self.nc_ids = pickle.loads(self.rcv())  # rcv nc ids
        LOGGER.debug(f"host get nc ids: {self.nc_ids}")
        for index, id in enumerate(self.nab_ids):
            if id in self.nc_ids:
                self.nc_indices.append(index)
        LOGGER.debug(f"nc indices: {self.nc_indices}")

    def __get_ub(self):
        self.ub = []
        self.ub_batchs = []
        batch_size = self.m_param.batch_size
        if batch_size == -1:
            batch_size = len(self.data_loader.data_frame)
        for i in range(0, len(self.nab_indices), batch_size):
            batch_start = i
            batch_end = batch_start + batch_size
            if batch_end > len(self.nab_indices):
                batch_end = len(self.nab_indices)
            x_batch = self.data_loader.data_matrix[
                self.nab_indices[batch_start:batch_end]
            ]

            # Net(x_batch)->ub_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ub_batch = self.forward(x_batch)
            self.ub += ub_batch
            self.ub_batchs.append(ub_batch)

        self.ub_nab = [self.ub[index] for index in self.nab_indices]
        # convert ub to numpy.array
        self.ub_nab_np = np.array([x.detach().numpy() for x in self.ub_nab])
        self.ub_nc_np = self.ub_nab_np[self.nc_indices]

    def __compute_hB(self):
        self.__get_ub()

        # The preceding None placeholder makes the subscript same as in the formula
        hB = [None] * 6

        hB[1] = self.ub_nc_np.T
        hB[2] = self.ub_nab_np.T
        hB[3] = None  # ignore the regular term
        hB[4] = self.m_param.const_gamma * \
            self.m_param.const_k * self.ub_nab_np
        hB[5] = np.expand_dims(self.m_param.const_gamma *
                               (self.ub_nab_np ** 2).sum(), axis=0)

        if self.m_param.mode == config.ENCRYPTED_MODE:
            for i in range(1, 6):
                hB[i] = self.encrypt(hB[i])
        return hB

    def __update_model(self, gradients):
        gradients = torch.tensor(gradients)
        self.backward(self.ub_nab, gradients_tensor=gradients)

    def __decrypt(self, en_matrix: np.array):
        if en_matrix is None:
            return None
        de_matrix = np.array(
            [self._private_key.decrypt(x)
             for x in en_matrix.flatten().tolist()]
        ).reshape(en_matrix.shape)
        return de_matrix

    @timer
    def __one_epoch(self):
        hB = self.__compute_hB()

        # send hB to guest
        self.send(pickle.dumps(hB))
        LOGGER.debug("host send hB to guest")

        noise_phi_ub, partial_ub_minus = pickle.loads(self.rcv())
        LOGGER.debug(
            "host received the middle part [[noise_phi_ub]] and [[partial_ub_minus]]"
        )

        if self.m_param.mode == config.ENCRYPTED_MODE:
            noise_phi_ub = self.__decrypt(noise_phi_ub)
            partial_ub_minus = self.__decrypt(partial_ub_minus)

        # compute the middle part
        middle_part = noise_phi_ub ** 2

        if self.m_param.mode == config.ENCRYPTED_MODE:
            middle_part = self.encrypt(middle_part)
        # send middle part to guest
        self.send(pickle.dumps(middle_part))
        LOGGER.debug("host send the middle part")

        # compute partial ub and update the model
        partial_ub = partial_ub_minus + 2 * self.m_param.const_gamma * self.ub_nab_np

        self.__update_model(gradients=partial_ub)

        # receive [[L]] and [[noised_partial_ua-]]
        L, noised_partial_ua_minus, noised_partial_ua_non = pickle.loads(
            self.rcv())
        LOGGER.debug(
            "host received [[L]], [[noised_partial_ua-]], [[noised_partial_ua_non]]"
        )

        if self.m_param.mode == config.ENCRYPTED_MODE:
            L, noised_partial_ua_minus, noised_partial_ua_non = (
                self.__decrypt(L),
                self.__decrypt(noised_partial_ua_minus),
                self.__decrypt(noised_partial_ua_non),
            )

        self.send(pickle.dumps(
            (L, noised_partial_ua_minus, noised_partial_ua_non)))
        LOGGER.debug("host send L and noised_partial_ua_minus")
        return L

    @timer
    def train(self):
        # check optimizer for the model
        if self._optimizer is None:
            self.set_optimizer()

        # send public key to guest
        self.send(pickle.dumps(self._public_key))
        LOGGER.debug("send public key to guest")

        LOGGER.info(f"host training, mode: {self.m_param.mode}")
        for epoch in tqdm.tqdm(range(self.m_param.epochs)):
            LOGGER.debug(f"-----epoch {epoch} begin-----")
            L = self.__one_epoch()
            LOGGER.debug(f"-----epoch {epoch} end, loss: {L}-----")
            self.predict()
            # receive stop signal
            signal = pickle.loads(self.rcv())
            LOGGER.debug(f"received signal: {signal}")
            if signal == config.END_SIGNAL:
                break

        LOGGER.info("end for training")

    def predict(self):
        LOGGER.info("predict begin...")
        LOGGER.debug("loading predict data")
        predict_data_loader = FTLDataLoader(self.m_param.predict_data_path)

        predict_ub_batchs = []
        batch_size = self.m_param.batch_size
        if batch_size == -1:
            batch_size = len(predict_data_loader.data_frame)
        for i in range(
            0, len(predict_data_loader.data_matrix), batch_size
        ):
            batch_start = i
            batch_end = batch_start + batch_size
            if batch_end > len(predict_data_loader.data_matrix):
                batch_end = len(predict_data_loader.data_matrix)
            x_batch = predict_data_loader.data_matrix[batch_start:batch_end]

            # Net(x_batch)->predict_ub_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            LOGGER.debug(f"x_batch: {x_batch}")
            predict_ub_batch = self.forward(x_batch)
            predict_ub_batchs += predict_ub_batch

        # convert to numpy.array
        predict_ub_batchs = np.array(
            [x.detach().numpy() for x in predict_ub_batchs])

        # send to guest
        self.send(pickle.dumps(predict_ub_batchs))
        LOGGER.debug("host send predict ubs")
        # receive results
        results, accuracy = pickle.loads(self.rcv())
        LOGGER.debug("host received predict results")
        return results, accuracy

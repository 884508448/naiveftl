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
        # generate key pair
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
        LOGGER.debug("computing hBs ...")
        self.__get_ub()

        # The preceding None placeholder makes the subscript same as in the formula
        hB = [None] * 6

        hB[1] = self.ub_nc_np.T
        hB[2] = self.ub_nab_np.T
        hB[3] = None  # ignore the regular term
        hB[4] = np.expand_dims(self.m_param.const_gamma *
                               (self.ub_nab_np ** 2).sum(), axis=0)
        hB[5] = np.array([np.expand_dims(
            ub_i, axis=1) @ np.expand_dims(ub_i, axis=0) for ub_i in self.ub_nc_np])

        if self.m_param.mode == config.ENCRYPTED_MODE:
            for i in range(1, 6):
                hB[i] = self.encrypt(hB[i])
        return hB

    def __update_model(self, gradients):
        gradients = torch.tensor(gradients)
        self.backward(self.ub_nab, gradients_tensor=gradients)

    @timer
    def __one_epoch(self):
        hB = self.__compute_hB()

        # send hB to guest
        self.send(pickle.dumps(hB))
        LOGGER.debug("host send hB to guest")

        # receive hA from guest
        LOGGER.debug("waitting for hA from guest")
        hA = pickle.loads(self.rcv())
        LOGGER.debug("host get hA from guest")

        # compute [[partial_ub]]
        LOGGER.debug("computing partial ub ...")
        h_partial_ub = hA[1]+self.ub_nc_np@hA[2] + \
            2*self.m_param.const_gamma*self.ub_nc_np

        # add mask for h_partial_ub
        masked_h_partial_ub = self.add_partial_mask(h_partial_ub)

        # receive masked_ua_partials and L, decrypt and send back
        LOGGER.debug("waitting for masked_ua_partials_nc, masked_ua_partials_non_overlap, L from guest")
        masked_ua_partials_nc, masked_ua_partials_non_overlap, L = pickle.loads(
            self.rcv())
        LOGGER.debug("masked_ua_partials_nc, masked_ua_partials_non_overlap, L received")
        if self.m_param.mode == config.ENCRYPTED_MODE:
            LOGGER.debug("decrypt masked_ua_partials and L")
            masked_ua_partials_nc = self.decrypt(masked_ua_partials_nc)
            masked_ua_partials_non_overlap = self.decrypt(
                masked_ua_partials_non_overlap)
            L = self.decrypt(L)
        LOGGER.debug("send masked_ua_partials L, and masked_h_partial_ub")
        self.send(pickle.dumps((masked_ua_partials_nc,
                  masked_ua_partials_non_overlap, L, masked_h_partial_ub)))

        # receive masked_partials
        LOGGER.debug("waitting for masked_partial_ub from guest")
        masked_partial_ub = pickle.loads(self.rcv())
        LOGGER.debug("masked_partial_ub received")

        # remove masks
        partial_ub = self.remove_partial_mask(masked_partial_ub)

        LOGGER.debug("updating model ...")
        self.__update_model(gradients=partial_ub)
        return L

    @timer
    def train(self):
        # check optimizer for the model
        if self._optimizer is None:
            self.set_optimizer()

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

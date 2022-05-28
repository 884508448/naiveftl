import pickle
import numpy as np
import torch
import tqdm

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_log import LOGGER
from phe import paillier
from utils import consts


class FTLHost(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLHost, self).__init__(param)
        self.nab_ids = None
        self.nab_indices = []  # nab indices refer to data_matrix
        self.nc_ids = None
        self.nc_indices = []  # nc indices refer to nab

        self.__get_overlap_id()
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

    def __get_ub(self):
        self.ub_nab = []
        self.ub_batchs=[]
        for i in range(0, len(self.nab_indices), self.m_param.batch_size):
            batch_start = i
            batch_end = batch_start + self.m_param.batch_size
            if batch_end > len(self.nab_indices):
                batch_end = len(self.nab_indices)
            x_batch = self.data_loader.data_matrix[
                self.nab_indices[batch_start:batch_end]
            ]

            # Net(x_batch)->ub_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ub_batch = self.forward(x_batch)
            self.ub_nab += ub_batch
            self.ub_batchs.append(ub_batch)

        # convert ub to numpy.array
        self.ub_nab_np = np.array([x.detach().numpy() for x in self.ub_nab])
        self.ub_nc = self.ub_nab_np[self.nc_indices]

    def __compute_hB(self):
        self.__get_ub()
        # The preceding None placeholder makes the subscript same as in the formula
        hB = [None] * 7

        hB[1] = self.ub_nc.T
        hB[2] = self.ub_nab_np.T
        hB[3] = None  # ignore the regular term
        hB[4] = (
            self.m_param.const_gamma
            * self.m_param.const_k
            * np.dot(np.ones(len(self.ub_nab_np)), self.ub_nab_np)
        )
        hB[5] = self.m_param.const_gamma * (self.ub_nab_np ** 2).sum()
        hB[6] = np.dot(np.ones(len(self.ub_nc)), self.ub_nc)
        return hB

    def __update_model(self, gradients):
        gradients = torch.tensor(gradients)
        for ub_batch in self.ub_batchs:
            self.backward(
                predicts=ub_batch, gradients_tensor=gradients
            )

    def __encrypt(self):
        pass

    def __decrypt(self):
        pass

    def train(self):

        # check optimizer for the model
        if self._optimizer is None:
            LOGGER.info(
                "optimizer has not been seted, it will be automatically seted as the default optimizer"
            )
            self.set_optimizer(
                optimizer=torch.optim.Adam(self._nn_model.parameters(), lr=0.01)
            )

        # generate key pair
        self._public_key, self._private_key = paillier.generate_paillier_keypair()
        # send public key to guest
        self.send(pickle.dumps(self._public_key))
        LOGGER.debug("send public key to guest")

        LOGGER.debug(f"host training, mode: {self.m_param.mode}")
        for epoch in tqdm.tqdm(range(self.m_param.epochs)):
            LOGGER.debug(f"-----epoch {epoch} begin-----")
            hB = self.__compute_hB()

            if self.m_param.mode == "plain":
                # send hB to guest
                self.send(pickle.dumps(hB))
                LOGGER.debug("host send hB to guest")

                noise_phi_ub, partial_ub_minus = pickle.loads(self.rcv())
                LOGGER.debug(
                    "host received the middle part phi_ub and encrypted partial ub-"
                )

                # expand into a matrix of the same shape as ub_nc
                phi_ub_matrix = np.expand_dims(noise_phi_ub, axis=0)
                phi_ub_matrix = phi_ub_matrix.repeat(len(self.ub_nc[0]), axis=0).T

                # compute the first middle part
                middle_part_1 = phi_ub_matrix * self.ub_nc

                # compute the second middle part
                middle_part_2 = noise_phi_ub ** 2

                # send middle parts to guest
                self.send(pickle.dumps((middle_part_1, middle_part_2)))
                LOGGER.debug("host send the two middle parts")

                # compute partial ub and update the model
                partial_ub = partial_ub_minus + 2 * self.m_param.const_gamma * np.dot(
                    np.ones(len(self.ub_nab_np)), self.ub_nab_np
                )

                self.__update_model(gradients=partial_ub)

                # receive encrypted L and noised partial ua-
                h_L, h_noised_partial_ua_minus = pickle.loads(self.rcv())
                LOGGER.debug("host received encrypted L and noised partial ua-")

                L, noised_partial_ua_minus = h_L, h_noised_partial_ua_minus
                self.send(pickle.dumps((L, noised_partial_ua_minus)))
                LOGGER.debug("host send L and noised partial ua-")

                # receive stop signal
                signal = pickle.loads(self.rcv())
                LOGGER.debug(f"received signal: {signal}")
                if signal == consts.END_SIGNAL:
                    LOGGER.info("end for training")
                    break


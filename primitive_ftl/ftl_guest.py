import numpy as np
import pickle
import torch
import tqdm

from phe import paillier
from common.ftl_base import FTLBase
from common.ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER
from utils import config
from utils.timer import timer


class FTLGuest(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLGuest, self).__init__(param)
        self.nab_ids = None  # N_AB ids
        self.nab_indices = []  # N_AB indices in dataset
        self.nc_ids = None  # N_c ids
        self.nc_indices = []  # N_c indices in dataset
        self.history_loss = []
        self.history_accu = []
        self._not_available_lable_ids = set()

        self.__get_overlap_ids()
        # generate key pair
        self._public_key, self._private_key = paillier.generate_paillier_keypair()
        LOGGER.info("ftl guest initialization finished")

    def __get_overlap_ids(self):
        """
        get nab, nc and the corresponding indices to these ids
        """

        host_ids = pickle.loads(self.rcv())  # rcv ids from host
        self.nab_ids = np.array(
            list(set(host_ids) & set(self.data_loader.id_index_map.keys()))
        )  # get intersection
        self.non_overlap_ids = np.array(
            list(set(self.data_loader.id_index_map.keys()) - set(self.nab_ids))
        )
        self.non_overlap_indices = np.array(
            [self.data_loader.id_index_map[id] for id in self.non_overlap_ids]
        )
        self.send(pickle.dumps(self.nab_ids))  # send intersection to host
        LOGGER.debug(f"guest get nab ids: {self.nab_ids}")

        # get nc_ids, nc_indices and corresponding indices for ids
        self.nc_ids = []
        for id in self.nab_ids:
            index = self.data_loader.id_index_map[id]
            self.nab_indices.append(index)
            if id not in self._not_available_lable_ids:
                self.nc_ids.append(id)
                self.nc_indices.append(index)
        self.send(pickle.dumps(self.nc_ids))
        LOGGER.debug(f"guest get nc ids: {self.nc_ids}")

    def __get_guest_components(self):
        # get middle results of guest: Φ_A, y, ua
        self.ua_batchs = []
        self.ua = []  # ua saves torch tensors
        self.phi_A = None
        self.y = np.array([])
        batch_size = self.m_param.batch_size
        if batch_size == -1:
            batch_size = len(self.data_loader.data_frame)
        for i in range(0, len(self.data_loader.data_frame), batch_size):
            batch_start = i
            batch_end = batch_start + batch_size
            if batch_end > len(self.data_loader.data_frame):
                batch_end = len(self.data_loader.data_frame)
            x_batch = self.data_loader.data_matrix[batch_start:batch_end]

            # Net(x_batch)->ua_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ua_batch = self.forward(x_batch)
            self.ua += ua_batch
            self.ua_batchs.append(ua_batch)

            # get y, compute Φ_A
            y_batch = self.data_loader.labels[batch_start:batch_end]
            self.y = np.concatenate((self.y, y_batch))
            phi_batch = np.dot(y_batch, ua_batch.detach().numpy())
            self.phi_A = self.phi_A + phi_batch if self.phi_A is not None else phi_batch

        self.phi_A /= len(self.data_loader.data_frame)
        self.ua_nab = [self.ua[index] for index in self.nab_indices]
        self.ua_nab_np = np.array([tensor.detach().numpy()
                                  for tensor in self.ua_nab])
        self.ua_non_overlap = [self.ua[index]
                               for index in self.non_overlap_indices]

    def __compute_guest_components(self):
        """
        compute and return all the parts of A that can be computed
        """

        LOGGER.debug("computing guest components ...")
        self.__get_guest_components()
        self.y_nc = self.y[self.nc_indices]
        self.y_non_overlap = self.y[self.non_overlap_indices]

        # convert ua to numpy array
        self.ua_np = np.array([x.detach().numpy() for x in self.ua])

        # get ua_nc
        self.ua_nc = self.ua_np[self.nc_indices]

        hA = [None]*3
        hA[1] = (
            -1
            / 2
            * np.dot(
                np.expand_dims(self.y_nc, axis=1), np.expand_dims(
                    self.phi_A, axis=0)
            )
            + self.m_param.const_gamma * self.m_param.const_k * self.ua_nc
        )

        hA[2] = 1 / 4 * np.expand_dims(self.phi_A,
                                       axis=1) @ np.expand_dims(self.phi_A, axis=0)

        L_part1 = np.array(len(self.y_nc) * np.log(2))
        L_part4 = self.m_param.const_gamma * np.sum(self.ua_nab_np ** 2)

        partial_ua_part3 = 2 * self.m_param.const_gamma * self.ua_nc

        if self.m_param.mode == config.ENCRYPTED_MODE:
            hA[1], hA[2] = (
                self.encrypt(hA[1]),
                self.encrypt(hA[2])
            )

        return hA, L_part1, L_part4, partial_ua_part3

    def __update_model(self, gradients, gradients_non_overlap):
        self.backward(
            predicts=self.ua_nab + self.ua_non_overlap,
            gradients_tensor=torch.tensor(
                np.concatenate([gradients, gradients_non_overlap])),
        )

    @timer
    def __one_epoch(self):
        (
            hA,
            L_part1,
            L_part4,
            partial_ua_part3,
        ) = self.__compute_guest_components()

        LOGGER.debug("waitting for hBs from host")
        hB = pickle.loads(self.rcv())
        LOGGER.debug("guest get hBs from host")

        # send hA to host
        self.send(pickle.dumps(hA))
        LOGGER.debug("guest send hA to host")

        phi_ub = np.dot(self.phi_A, hB[1])
        L_part2 = -1 / 2 * np.dot(np.expand_dims(self.y_nc, axis=0), phi_ub)
        L_part3 = 0
        for matrix in hB[5]:
            L_part3 = L_part3 + \
                np.expand_dims(self.phi_A, axis=0) @ matrix @ np.expand_dims(self.phi_A, axis=1)
        L_part3 = L_part3/8
        L_part5 = hB[4]
        L_part6 = np.array(
            self.m_param.const_gamma
            * self.m_param.const_k
            * (self.ua_nab_np @ hB[2]).sum()
        )

        h_L = (L_part1 + L_part2 + L_part3 + L_part4 + L_part5 + L_part6) / len(
            self.ua_nab_np
        )
        LOGGER.debug(f"h_L shape: {h_L.shape}")

        # partial ua nc
        y_ub_sum = np.expand_dims(self.y_nc, axis=0) @ hB[1].T
        LOGGER.debug(f"y_ub_sum shape:{y_ub_sum.shape}")
        partial_ua_part1 = (
            -0.5
            / len(self.y)
            * np.expand_dims(self.y_nc, axis=1) @ y_ub_sum

        )
        LOGGER.debug(f"partial_ua_part1 shape:{partial_ua_part1.shape}")
        ub_matrix_sum = np.sum(np.array(hB[5]), axis=0)
        LOGGER.debug(f"ub_matrix_sum shape:{ub_matrix_sum.shape}")
        partial_ua_part2 = (
            0.25
            / len(self.y)
            * np.expand_dims(self.y_nc, axis=1) @ np.expand_dims(self.phi_A, axis=0) @ ub_matrix_sum
        )
        LOGGER.debug(f"partial_ua_part2 shape{partial_ua_part2.shape}")
        partial_ua_part4 = self.m_param.const_gamma * \
            self.m_param.const_k*hB[1].T
        LOGGER.debug(f"partial_ua_part4 shape{partial_ua_part4.shape}")
        h_partial_ua_nc = partial_ua_part1 + partial_ua_part2 + \
            partial_ua_part3 + partial_ua_part4

        # partial ua non overlap
        h_partial_ua_part1_non = (
            -0.5
            / len(self.y)
            * np.expand_dims(self.y_non_overlap, axis=1) @ y_ub_sum
        )

        h_partial_ua_part2_non = (
            0.25
            / len(self.y)
            * np.expand_dims(self.y_non_overlap, axis=1)
            @ np.expand_dims(self.phi_A, axis=0)
            @ ub_matrix_sum
        )
        h_partial_ua_non = h_partial_ua_part1_non + h_partial_ua_part2_non

        # add mask for partial_ua_nc
        masked_h_partial_ua_nc = self.add_partial_mask(h_partial_ua_nc)
        # add mask for partial_ua_non
        masked_h_partial_ua_non_overlap = self.add_partial_mask_non_overlap(
            h_partial_ua_non)

        # send masked partials and [[L]] to host
        LOGGER.debug("send masked partials and [[L]] to host")
        self.send(pickle.dumps((masked_h_partial_ua_nc,
                  masked_h_partial_ua_non_overlap, h_L)))

        # receive masked_partials and L
        LOGGER.debug("waitting for masked_ua_partials L, and masked_h_partial_ub from host")
        masked_partial_ua, masked_partial_ua_non, L, masked_ub_partials = pickle.loads(
            self.rcv())

        if self.m_param.mode == config.ENCRYPTED_MODE:
            LOGGER.debug("decrypt masked_h_ub_partials ...")
            masked_ub_partials = self.decrypt(masked_ub_partials)
        LOGGER.debug("send masked ub partials back")
        self.send(pickle.dumps(masked_ub_partials))

        # remove masks
        partial_ua, partial_ua_non = self.remove_partial_mask(
            masked_partial_ua), self.remove_partial_mask_non_overlap(masked_partial_ua_non)

        LOGGER.debug("updating model ...")
        LOGGER.debug(
            f"partial_ua shape:{partial_ua.shape}, partial_ua_non shape:{partial_ua_non.shape}")
        self.__update_model(partial_ua, partial_ua_non)
        return L

    @timer
    def train(self):
        # check optimizer for the model
        if self._optimizer is None:
            self.set_optimizer()

        LOGGER.info(f"guest training, mode: {self.m_param.mode}")
        for epoch in tqdm.tqdm(range(self.m_param.epochs)):
            LOGGER.debug(f"-----epoch {epoch} begin-----")
            L = self.__one_epoch()
            self.history_loss.append(float(L))
            LOGGER.debug(f"-----epoch {epoch} end, loss: {L}-----")
            self.predict()
            if L < self.m_param.loss_tol:
                LOGGER.info(
                    f"the loss {L} is smaller than the loss tolerance {self.m_param.loss_tol}"
                )
                self.send(pickle.dumps(config.END_SIGNAL))
                LOGGER.debug(f"send signal: {config.END_SIGNAL}")
                break

            else:
                self.send(pickle.dumps(config.CONTINUE_SIGNAL))
                LOGGER.debug(f"send signal: {config.CONTINUE_SIGNAL}")

        LOGGER.info("end for training")
        LOGGER.debug(f"loss history: {self.history_loss}")
        LOGGER.debug(f"history accuracy: {self.history_accu}")

    def set_not_available_lables(self, na_set):
        """
        the labels given in na_set are not availabel for training,
        which means N_AB-na_set=N_c
        """

        self._not_available_lable_ids = na_set
        LOGGER.info(
            f"the tags corresponding to these ids are not available for training: {na_set}"
        )

    def predict(self):
        # init
        # load data
        predic_data_loader = FTLDataLoader(self.m_param.predict_data_path)
        labels = predic_data_loader.labels
        # get predict_phi_A
        predict_phi_A = None
        batch_size = self.m_param.batch_size
        if batch_size == -1:
            batch_size = len(predic_data_loader.data_frame)
        for i in range(0, len(predic_data_loader.data_frame), batch_size):
            batch_start = i
            batch_end = batch_start + batch_size
            if batch_end > len(predic_data_loader.data_frame):
                batch_end = len(predic_data_loader.data_frame)
            x_batch = predic_data_loader.data_matrix[batch_start:batch_end]
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ua_batch = self.forward(x_batch)
            y_batch = labels[batch_start:batch_end]
            phi_A_batch = np.dot(y_batch, ua_batch.detach().numpy())
            predict_phi_A = (
                predict_phi_A + phi_A_batch
                if predict_phi_A is not None
                else phi_A_batch
            )
        predict_phi_A /= len(predic_data_loader.data_frame)

        # receive predict ubs from host
        LOGGER.debug("waitting for predict ubs from host")
        predict_ubs = pickle.loads(self.rcv())
        LOGGER.debug("predict ubs received")
        results = np.dot(predict_phi_A, predict_ubs.T)

        # compute accuracy
        correct_num = 0
        positive_num = 0
        LOGGER.debug(f"results: {results}")
        LOGGER.debug(f"labels: {labels}")
        assert len(results) == len(
            labels
        ), "the results length not equal to lables length"
        for i in range(len(results)):
            if results[i] > 0:
                positive_num += 1
            if results[i] > 0 and labels[i] == 1 or results[i] < 0 and labels[i] == -1:
                correct_num += 1
        accuracy = correct_num / len(results)
        self.history_accu.append(accuracy)
        # send results, accuracy to host
        self.send(pickle.dumps((results, accuracy)))
        LOGGER.debug("guest send predict results")
        LOGGER.debug(
            f"positive num: {positive_num}, negative num: {len(results)-positive_num}"
        )

import numpy as np
import pickle
import torch
import tqdm

from ftl_base import FTLBase
from ftl_param import FTLParam
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

        self.__get_guest_components()
        self.y_nc = self.y[self.nc_indices]
        self.y_non_overlap = self.y[self.non_overlap_indices]

        # convert ua to numpy array
        self.ua_np = np.array([x.detach().numpy() for x in self.ua])

        # get ua_nc
        self.ua_nc = self.ua_np[self.nc_indices]

        h1_A = (
            -1
            / 2
            * np.dot(
                np.expand_dims(self.y_nc, axis=1), np.expand_dims(
                    self.phi_A, axis=0)
            )
            + self.m_param.const_gamma * self.m_param.const_k * self.ua_nc
        )

        h2_A = 1 / 4 * np.dot(self.phi_A, self.phi_A)

        L_part1 = np.array(len(self.y_nc) * np.log(2))
        L_part4 = self.m_param.const_gamma * np.sum(self.ua_nab_np ** 2)

        partial_ua_part4 = 2 * self.m_param.const_gamma * self.ua_nc

        if self.m_param.mode == config.ENCRYPTED_MODE:
            h1_A, L_part1, L_part4 = (
                self.encrypt(h1_A),
                self.encrypt(L_part1),
                self.encrypt(L_part4),
            )

        return h1_A, h2_A, L_part1, L_part4, partial_ua_part4

    def __add_noise_ma1(self, phi_ub):
        """
        return the result of nised phi_ub
        and reserve the masks
        """

        self._ma1 = np.random.random(len(self.y_nc))
        return self._ma1 * phi_ub

    def __remove_noise_ma1(self, noise_2_data):
        reci_ma1 = np.ones_like(self._ma1) / self._ma1
        reci_ma1_2 = reci_ma1 ** 2

        return noise_2_data * reci_ma1_2

    def __add_noise_ma2(self, partial_ua_minus):
        self._ma2 = np.random.random(size=partial_ua_minus.shape)
        return self._ma2 + partial_ua_minus

    def __remove_noise_ma2(self, noise_data):
        return noise_data - self._ma2

    def __add_noise_ma3(self, partial_ua_non):
        self._ma3 = np.random.random(size=partial_ua_non.shape)
        return self._ma3 + partial_ua_non

    def __remove_noise_ma3(self, noise_data):
        return noise_data - self._ma3

    def __update_model(self, gradients, gradients_non):
        self.backward(
            predicts=self.ua_nab + self.ua_non_overlap,
            gradients_tensor=torch.tensor(
                np.concatenate([gradients, gradients_non])),
        )

    @timer
    def __one_epoch(self):
        (
            h1_A,
            h2_A,
            L_part1,
            L_part4,
            partial_ua_part4,
        ) = self.__compute_guest_components()

        hB = pickle.loads(self.rcv())
        LOGGER.debug("guest get hBs from host")

        # compute and send the middle part
        phi_ub = np.dot(self.phi_A, hB[1])
        noise_phi_ub = self.__add_noise_ma1(phi_ub)

        # compute partial_ub-
        partial_ub_part2 = h2_A * hB[1].T
        partial_ub_minus = h1_A + partial_ub_part2

        self.send(pickle.dumps((noise_phi_ub, partial_ub_minus)))
        LOGGER.debug(
            "guest send the middle part [[noise_phi_ub]] and partial [[ub-]]")

        # receive middle part from host
        noise_ma1_2_data = pickle.loads(self.rcv())
        LOGGER.debug("guest received the middle part")
        middle = self.__remove_noise_ma1(noise_ma1_2_data)

        L_part2 = -1 / 2 * np.dot(np.expand_dims(self.y_nc, axis=0), phi_ub)
        L_part5 = hB[5]
        L_part6 = np.array(
            self.m_param.const_gamma
            * self.m_param.const_k
            * (self.ua_nab_np * hB[2].T).sum()
        )

        L_part3 = 1 / 8 * \
            np.dot(np.expand_dims(np.ones_like(middle), axis=0), middle)

        h_L = (L_part1 + L_part2 + L_part3 + L_part4 + L_part5 + L_part6) / len(
            self.ua_nab_np
        )

        # partial ua nc
        y_ub_sum = np.dot(self.y_nc, hB[1].T)
        partial_ua_part1 = (
            -0.5
            / len(self.y)
            * np.dot(
                np.expand_dims(self.y_nc, axis=1), np.expand_dims(
                    y_ub_sum, axis=0)
            )
        )
        partial_ua_part3 = hB[4]
        partial_ua_part2 = (
            0.25
            / len(self.y)
            * hB[5]
            / self.m_param.const_gamma
            * np.dot(
                np.expand_dims(self.y_nc, axis=1), np.expand_dims(
                    self.phi_A, axis=0)
            )
        )
        h_partial_ua_minus = partial_ua_part1 + partial_ua_part2 + partial_ua_part3

        # partial ua non overlap
        h_partial_ua_part1_non = (
            -0.5
            / len(self.y)
            * np.dot(
                np.expand_dims(self.y_non_overlap, axis=1),
                np.expand_dims(y_ub_sum, axis=0),
            )
        )
        h_partial_ua_part2_non = (
            0.25
            / len(self.y)
            * hB[5]
            / self.m_param.const_gamma
            * np.dot(
                np.expand_dims(self.y_non_overlap, axis=1),
                np.expand_dims(self.phi_A, axis=0),
            )
        )
        h_partial_ua_non = h_partial_ua_part1_non + h_partial_ua_part2_non

        # noise partial_ua -
        h_noised_partial_ua_minus = self.__add_noise_ma2(h_partial_ua_minus)
        h_noised_partial_ua_non = self.__add_noise_ma3(h_partial_ua_non)
        # send the [[L]], [[noised_partial_ua-]], [[noised_partial_ua_non]] to host
        self.send(pickle.dumps(
            (h_L, h_noised_partial_ua_minus, h_noised_partial_ua_non)))
        LOGGER.debug(
            "guest send [[L]], [[noised_partial_ua-]], [[noised_partial_ua_non]]")

        # receive L, noised partial_ua-
        L, noised_partial_ua_minus, noised_partial_ua_non = pickle.loads(
            self.rcv())
        LOGGER.debug(
            "guest received L, noised_partial_ua_minus, noised_partial_ua_non")

        # compute partial_ua and update model
        partial_ua_minus = self.__remove_noise_ma2(noised_partial_ua_minus)
        partial_ua_non = self.__remove_noise_ma3(noised_partial_ua_non)
        partial_ua = partial_ua_minus + partial_ua_part4

        self.__update_model(partial_ua, partial_ua_non)
        return L

    @timer
    def train(self):
        # check optimizer for the model
        if self._optimizer is None:
            self.set_optimizer()

        # receive public key from host
        self._public_key = pickle.loads(self.rcv())
        LOGGER.debug("receive public key from host")

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

import numpy as np
import pickle
import torch
import tqdm

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER
from utils import consts


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
        for i in range(0, len(self.data_loader.data_frame), self.m_param.batch_size):
            batch_start = i
            batch_end = batch_start + self.m_param.batch_size
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
        self.display("ua",self.ua)
        self.display("phi_A",self.phi_A)
        self.display("y",self.y)

    def __compute_guest_components(self):
        """
        compute and return all the parts of A that can be computed
        """

        self.__get_guest_components()
        self.y_nc = self.y[self.nc_indices]
        self.display("y nc",self.y_nc)

        # convert ua to numpy array
        self.ua_np = np.array([x.detach().numpy() for x in self.ua])

        # get ua_nab and ua_nc
        self.ua_nab = self.ua_np[self.nab_indices]
        self.ua_nc = self.ua_np[self.nc_indices]
        self.display("ua nc",self.ua_nc)

        h1_A = -1/2 * self.phi_A * np.dot(
            self.y_nc, np.ones_like(self.y_nc)
        ) + self.m_param.const_gamma * self.m_param.const_k * np.dot(
            np.ones(len(self.ua_nab)), self.ua_nab
        )
        self.display("h1_A",h1_A)

        h2_A = 1/4 * np.dot(self.phi_A, self.phi_A)
        self.display("h2_A",h2_A)

        L_part1 = len(self.y_nc) * np.log(2)
        self.display("L_part1",L_part1)
        L_part4 = self.m_param.const_gamma * np.sum(self.ua_nab ** 2)
        self.display("L_part4",L_part4)

        ua_nab_sum = np.dot(np.ones(len(self.ua_nab)), self.ua_nab)

        partial_ua_part4 = 2 * self.m_param.const_gamma * ua_nab_sum
        self.display("partial_ua_part4",partial_ua_part4)

        return h1_A, h2_A, L_part1, L_part4, partial_ua_part4

    def __add_noise_ma1(self, phi_ub):
        """
        return the result of nised phi_ub
        and reserve the masks
        """

        self._ma1 = np.random.random(len(self.y_nc))
        return self._ma1 * phi_ub

    def __remove_noise_ma1(self, noise_data, noise_2_data):
        reci_ma1 = np.ones_like(self._ma1) / self._ma1
        reci_ma1_2 = reci_ma1 ** 2

        # convert reci_ma1 to the same shape as noise_data
        reci_ma1 = np.expand_dims(reci_ma1, axis=0)
        reci_ma1 = reci_ma1.repeat(len(self.ua_np[0]), axis=0).T

        return noise_data * reci_ma1, noise_2_data * reci_ma1_2

    def __add_noise_ma2(self, partial_ua_minus):
        self._ma2 = np.random.random(len(partial_ua_minus))
        return self._ma2 + partial_ua_minus

    def __remove_noise_ma2(self, noise_data):
        return noise_data - self._ma2

    def __update_model(self, gradients):
        gradients = torch.tensor(gradients)
        for ua_batch in self.ua_batchs:
            self.backward(predicts=ua_batch, gradients_tensor=gradients)

    def train(self):

        # check optimizer for the model
        if self._optimizer is None:
            LOGGER.info(
                "optimizer hs not been seted, it will be automatically seted as the default optimizer"
            )
            self.set_optimizer(
                optimizer=torch.optim.Adam(self._nn_model.parameters(), lr=self.m_param.learning_rate)
            )

        # receive public key from host
        self.host_public_key = pickle.loads(self.rcv())
        LOGGER.debug("receive public key from host")

        LOGGER.info(f"guest training, mode: {self.m_param.mode}")
        for epoch in tqdm.tqdm(range(self.m_param.epochs)):
            LOGGER.info(f"-----epoch {epoch} begin-----")
            (
                h1_A,
                h2_A,
                L_part1,
                L_part4,
                partial_ua_part4,
            ) = self.__compute_guest_components()

            hB = pickle.loads(self.rcv())
            LOGGER.debug("guest get hBs from host")

            if self.m_param.mode == consts.PLAIN_MODE:
                # compute and send the middle part
                phi_ub = np.dot(self.phi_A, hB[1])
                noise_phi_ub = self.__add_noise_ma1(phi_ub)
                self.display("phi_ub",phi_ub)
                self.display("noise_phi_ub",noise_phi_ub)

                # compute partial_ub-
                partial_ub_part2 = h2_A * hB[6]
                partial_ub_minus = h1_A + partial_ub_part2
                self.display("partial_ub_part2",partial_ub_part2)
                self.display("partial_ub_minus",partial_ub_minus)

                self.send(pickle.dumps((noise_phi_ub, partial_ub_minus)))
                LOGGER.debug("guest send the middle part phi_ub and partial ub-")

                # receive two middle part from host
                noise_ma1_data, noise_ma1_2_data = pickle.loads(self.rcv())
                LOGGER.debug("guest received the two middle parts")
                middle1, middle2 = self.__remove_noise_ma1(
                    noise_ma1_data, noise_ma1_2_data
                )
                self.display("middle1",middle1)
                self.display("middle2",middle2)

                L_part2 = -1/2 * np.dot(self.y_nc, phi_ub)
                L_part5 = hB[5]
                L_part6 = (
                    self.m_param.const_gamma
                    * self.m_param.const_k
                    * (self.ua_nab * hB[2].T).sum()
                )
                self.display("L_part2",L_part2)
                self.display("L_part5",L_part5)
                self.display("L_part6",L_part6)

                partial_ua_part1 = -1/2 / len(self.ua) * hB[6]
                partial_ua_part3 = hB[4]
                self.display("partial_ua_part1",partial_ua_part1)
                self.display("partial_ua_part3",partial_ua_part3)

                L_part3 = 1/8 * np.dot(np.ones_like(middle2), middle2)
                self.display("L_part3",L_part3)

                h_L = (L_part1 + L_part2 + L_part3 + L_part4 + L_part5 + L_part6)/len(self.ua_nab)
                self.display("h_L",h_L)
                # debug
                # h_L = (L_part1 + L_part2 + L_part3 + L_part6)/len(self.ua_nab)

                partial_ua_part2 = 1/4 / len(self.ua) * np.dot(self.y_nc, middle1)
                h_partial_ua_minus = (
                    partial_ua_part1 + partial_ua_part2 + partial_ua_part3
                )
                self.display("partial_ua_part2",partial_ua_part2)
                self.display("h_partial_ua_minus",h_partial_ua_minus)

                # noise partial_ua -
                noised_partial_ua_minus = self.__add_noise_ma2(h_partial_ua_minus)
                # send the encrypted L, noised partial_ua- to host
                self.send(pickle.dumps((h_L, noised_partial_ua_minus)))
                LOGGER.debug("guest send encrypted L and noised partial ua")

                # receive L, noised partial_ua-
                L, noised_partial_ua_minus = pickle.loads(self.rcv())
                LOGGER.debug("guest received L and noised partial ua")

                # compute partial_ua and update model
                partial_ua_minus = self.__remove_noise_ma2(noised_partial_ua_minus)
                partial_ua = partial_ua_minus + partial_ua_part4
                # debug
                # partial_ua = partial_ua_minus
                partial_ua /= len(self.ua_nab)
                self.display("partial_ua",partial_ua)
                self.__update_model(partial_ua)

            else:
                ...

            self.history_loss.append(L)


            LOGGER.info(f"-----epoch {epoch} end, loss: {L}-----")
            if L < self.m_param.loss_tol:
                LOGGER.info(
                    f"the loss {L} is smaller than the loss tolerance {self.m_param.loss_tol}"
                )
                self.send(pickle.dumps(consts.END_SIGNAL))
                LOGGER.debug(f"send signal: {consts.END_SIGNAL}")
                break

            else:
                self.send(pickle.dumps(consts.CONTINUE_SIGNAL))
                LOGGER.debug(f"send signal: {consts.CONTINUE_SIGNAL}")
            
            self.predict()

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
        for i in range(0, len(predic_data_loader.data_frame), self.m_param.batch_size):
            batch_start = i
            batch_end = batch_start+self.m_param.batch_size
            if batch_end > len(predic_data_loader.data_frame):
                batch_end = len(predic_data_loader.data_frame)
            x_batch = predic_data_loader.data_matrix[batch_start:batch_end]
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ua_batch = self.forward(x_batch)
            y_batch = labels[batch_start:batch_end]
            phi_A_batch = np.dot(y_batch,ua_batch.detach().numpy())
            predict_phi_A = predict_phi_A+phi_A_batch if predict_phi_A is not None else phi_A_batch
        predict_phi_A/=len(predic_data_loader.data_frame)

        # receive predict ubs from host
        predict_ubs = pickle.loads(self.rcv())
        LOGGER.debug("predict ubs received")
        results = np.dot(predict_phi_A, predict_ubs.T)

        # compute accuracy
        correct_num = 0
        positive_num = 0
        assert len(results) == len(
            labels
        ), "the results length not equal to lables length"
        for i in range(len(results)):
            if results[i] > 0:
                positive_num +=1
            if results[i] > 0 and labels[i] == 1 or results[i] < 0 and labels[i] == -1:
                correct_num += 1
        accuracy = correct_num / len(results)
        self.history_accu.append(accuracy)
        # send results, accuracy to host
        self.send(pickle.dumps((results, accuracy)))
        LOGGER.debug("guest send predict results")
        LOGGER.debug(f"positive num: {positive_num}, negative num: {len(results)-positive_num}")


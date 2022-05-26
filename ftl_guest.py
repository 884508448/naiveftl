import numpy as np
import pickle
import torch

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_log import LOGGER


class FTLGuest(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLGuest, self).__init__(param)
        self.phi = None  # Φ_A
        self.overlap_id = None
        self.overlap_indices = []
        self.history_loss = []

        self.__get_overlap_id()
        LOGGER.info("ftl guest initialization finished")

    def __get_overlap_id(self):
        host_ids = pickle.loads(self.rcv())  # rcv ids from host
        self.overlap_id = np.array(
            list(set(host_ids) & set(self.data_loader.id_index_map.keys()))
        )  # get intersection
        self.send(pickle.dumps(self.overlap_id))  # send intersection to host
        LOGGER.debug(f"guest get overlap id: {self.overlap_id}")
        for id in self.overlap_id:
            self.overlap_indices.append(self.data_loader.id_index_map[id])

    def __get_guest_components(self):
        # get middle results of guest: Φ_A, y, ua
        ua = []
        phi_A = None
        y = []
        for i in range(0, len(self.data_loader.data_frame), self.m_param.batch_size):
            batch_start = i
            batch_end = batch_start + self.m_param.batch_size
            if batch_end > len(self.data_loader.data_frame):
                batch_end = len(self.data_loader.data_frame)
            x_batch = self.data_loader.data_matrix[batch_start:batch_end]

            # Net(x_batch)->ua_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ua_batch = self.forward(x_batch)
            ua += ua_batch

            # get y, compute Φ_A
            y_batch = self.data_loader.labels[batch_start:batch_end]
            y += y_batch
            phi_batch = np.dot(y_batch, ua_batch.detach().numpy())
            phi_A = phi_A + phi_batch if phi_A is not None else phi_batch

        phi_A /= len(self.data_loader.data_frame)
        return ua, phi_A, y

    def compute_guest_components(self):
        h1 = -0.5 * np.matmul(self.overlap_y, self.overlap_ua)
        LOGGER.debug(f"h1 (a vector): {h1}")
        h2 = 0.25 * np.matmul(self.phi, self.phi.T)
        LOGGER.debug(f"h2 (a number): {h2}")
        h3 = (
            self.m_param.const_lambda
            * self.m_param.const_k
            * np.matmul(
                np.ones((1, len(self.overlap_id)), dtype=np.float), self.overlap_ua
            )
        )
        LOGGER.debug(f"h3 (a vector): {h3}")
        self.send_components = h1, h2, h3

    def compute_loss(self):
        # hb_1, hb_2, hb_4, hb_5 = self.get_components
        # part1 = len(self.overlap_id) * np.log(2)
        # part2 = -0.5*np.matmul( ,np.matmul(self.phi,hb_1).T)
        pass

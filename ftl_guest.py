import numpy as np
import pickle

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_log import LOGGER


class FTLGuest(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLGuest, self).__init__(param)
        self.phi = None  # Φ_A
        self.overlap_id = None
        self.overlap_y = None  # y_i ∈ N_c
        self.overlap_ua = None  # u_i ∈ N_AB
        self.send_components = None  # components to send: hA_1~3
        self.get_components = None  # components get from host: hB_1,4,5
        self.history_loss = []
        self.__init_params()
        LOGGER.info("ftl guest initialization finished")

    def __init_params(self):
        self.__get_overlap_id()
        self.phi = (1 / len(self.data_loader.labels)) * np.matmul(self.data_loader.labels,
                                                                  self.data_loader.data_matrix)  # phi = 1/N_A*Sigma(yi*u_i)
        self.overlap_y = self.data_loader.labels[self.overlap_id]
        self.overlap_ua = self.data_loader.data_matrix[self.overlap_id]

    def __get_overlap_id(self):
        host_ids = pickle.loads(self.rcv())  # rcv ids from host
        self.overlap_id = np.array(list(set(host_ids) & set(self.data_loader.id_index_map.keys())))  # get intersection
        self.send(pickle.dumps(self.overlap_id))  # send intersection to host
        LOGGER.debug(f"guest get overlap id: {self.overlap_id}")

    def compute_guest_components(self):
        h1 = -0.5 * np.matmul(self.overlap_y, self.overlap_ua)
        LOGGER.debug(f"h1 (a vector): {h1}")
        h2 = 0.25 * np.matmul(self.phi, self.phi.T)
        LOGGER.debug(f"h2 (a number): {h2}")
        h3 = self.m_param.const_lambda * self.m_param.const_k * np.matmul(
            np.ones((1, len(self.overlap_id)), dtype=np.float), self.overlap_ua)
        LOGGER.debug(f"h3 (a vector): {h3}")
        self.send_components = h1, h2, h3

    def compute_loss(self):
        # hb_1, hb_2, hb_4, hb_5 = self.get_components
        # part1 = len(self.overlap_id) * np.log(2)
        # part2 = -0.5*np.matmul( ,np.matmul(self.phi,hb_1).T)
        pass
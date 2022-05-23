import pickle
import numpy as np

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_log import LOGGER


class FTLHost(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLHost, self).__init__(param)
        self.overlap_id = None
        self.overlap_ub = None  # u_b
        self.send_components = None  # components to send: hB_1,2,4,5
        self.get_components = None  # components get from guest: hA_1~3

        self.__init_param()
        LOGGER.info("ftl host initialization finished")

    def __init_param(self):
        self.__get_overlap_id()
        self.overlap_ub = self.data_loader.data_matrix[self.overlap_id]

    def __get_overlap_id(self):
        self.send(pickle.dumps(list(self.data_loader.id_index_map.keys())))  # send ids to guest
        self.overlap_id = pickle.loads(self.rcv())  # rcv overlap ids
        LOGGER.debug(f"host get overlap id: {self.overlap_id}")

    def compute_host_components(self):
        h1 = self.overlap_ub.T
        LOGGER.debug(f"h1 (a matrix): {h1}")
        h4 = np.array([float(np.matmul(self.overlap_ub[i], self.overlap_ub[i].T)) for i in range(len(self.overlap_id))])
        LOGGER.debug(f"h4 (a vector): {h4}")
        h5 = self.m_param.const_lambda * self.m_param.const_k * np.matmul(
            np.ones((1, len(self.overlap_id)), dtype=np.float), self.overlap_ub)
        LOGGER.debug(f"h5 (a vector) :{h5}")
        self.send_components = h1, h4, h5

    def encrypt(self):
        pass

    def decrypt(self):
        pass
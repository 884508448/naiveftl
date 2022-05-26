import pickle
import numpy as np
import torch

from ftl_base import FTLBase
from ftl_param import FTLParam
from utils.ftl_log import LOGGER


class FTLHost(FTLBase):
    def __init__(self, param: FTLParam):
        super(FTLHost, self).__init__(param)
        self.overlap_id = None
        self.overlap_indices=[]

        self.__get_overlap_id()
        LOGGER.info("ftl host initialization finished")

    def __get_overlap_id(self):
        self.send(pickle.dumps(list(self.data_loader.id_index_map.keys())))  # send ids to guest
        self.overlap_id = pickle.loads(self.rcv())  # rcv overlap ids
        LOGGER.debug(f"host get overlap id: {self.overlap_id}")
        for id in self.overlap_id:
            self.overlap_indices.append(self.data_loader.id_index_map[id])

    def __get_ub(self):
        ub=[]
        for i in range(0, len(self.data_loader.data_frame), self.m_param.batch_size):
            batch_start = i
            batch_end = batch_start + self.m_param.batch_size
            if batch_end > len(self.data_loader.data_frame):
                batch_end = len(self.data_loader.data_frame)
            x_batch = self.data_loader.data_matrix[batch_start:batch_end]

            # Net(x_batch)->ub_batch
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            ub_batch = self.forward(x_batch)
            ub += ub_batch
        return ub

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
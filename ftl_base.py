import socket
import utils.consts as consts
import torch

from ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER


class FTLBase:
    def __init__(self, param: FTLParam):
        self.m_param = param
        self._nn_model = torch.nn.Sequential()
        self._layer_index = 0
        self._optimizer = torch.optim.Adam(self._nn_model.parameters(), lr=0.01)
        LOGGER.info(f"ftl {self.m_param.role} starting")

        LOGGER.debug("loading data")
        self.data_loader = FTLDataLoader(self.m_param.data_path)
        if self.m_param.batch_size == -1:
            self.m_param.batch_size = len(self.data_loader.data_frame)

        LOGGER.info("building connection")
        self.__init_socket()

    def __init_socket(self):
        self.m_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.m_param.role == consts.GUEST:
            self.m_sock.bind((consts.DEFAULT_IP, consts.GUEST_DEFAULT_PORT),)
            self.m_sock.listen()
            conn, addr = self.m_sock.accept()
            assert (
                addr == self.m_param.partner_addr
            ), f"get connection from: {addr}, not equal to offered: {self.m_param.partner_addr}"
            self.__messenger: socket.socket = conn
        else:
            self.m_sock.bind((consts.DEFAULT_IP, consts.HOST_DEFAULT_PORT))
            self.m_sock.connect(self.m_param.partner_addr)
            self.__messenger: socket.socket = self.m_sock

    def send(self, msg):
        # send msg to partner
        self.__messenger.sendall(msg)

    def rcv(self):
        # receive msg from partner, the buffer size is defined in consts.py
        return self.__messenger.recv(consts.DEFAULT_BUFFER_SIZE)

    def add_nn_layer(self, layer):
        self._nn_model.add_module(name=f"layer {self._layer_index}", module=layer)
        self._layer_index += 1
        LOGGER.debug(f"add layer {layer} successfully")

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim)
        old_optimizer = self._optimizer
        self._optimizer = optimizer
        LOGGER.debug(
            f"set optimizer successfully, optimizer changed: {old_optimizer} -> {self._optimizer}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nn_model(x)

    def backward(self, predicts: torch.Tensor, gradients_tensor: torch.Tensor):
        self._optimizer.zero_grad()
        predicts.backward(gradient=gradients_tensor)
        self._optimizer.step()

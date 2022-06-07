import socket
import utils.consts as consts
import torch
import pickle
import time

from ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER


class FTLBase:
    def __init__(self, param: FTLParam):
        self.m_param = param
        self._nn_model = torch.nn.Sequential()
        self._layer_index = 0
        self._optimizer = None
        LOGGER.info(f"ftl {self.m_param.role} starting")

        LOGGER.debug("loading data")
        self.data_loader = FTLDataLoader(self.m_param.data_path)
        if self.m_param.batch_size == -1:
            self.m_param.batch_size = len(self.data_loader.data_frame)

        LOGGER.debug("building connection")
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
            self._messenger: socket.socket = conn
        else:
            self.m_sock.bind((consts.DEFAULT_IP, consts.HOST_DEFAULT_PORT))
            self.m_sock.connect(self.m_param.partner_addr)
            self._messenger: socket.socket = self.m_sock
        # self._messenger.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,True)
        LOGGER.debug("connection builded")

    def send(self, msg: bytes):
        # send msg length to partner
        msg_length=len(msg)
        self._messenger.sendall(pickle.dumps(msg_length))
        LOGGER.debug(f"send msg length: {msg_length}")
        # sending too fast can cause occasional packet loss, the same as below
        time.sleep(consts.NET_DELAY)

        # avoid continuous send
        received_msg_length = pickle.loads(self._messenger.recv(consts.DEFAULT_BUFFER_SIZE))
        assert received_msg_length == msg_length
        LOGGER.debug(f"assert msg length: {msg_length}")
        time.sleep(consts.NET_DELAY)

        # send msg to partner
        self._messenger.sendall(msg)
        LOGGER.debug(f"send msg")
        time.sleep(consts.NET_DELAY)

        # avoid continuous send
        received_length = pickle.loads(self._messenger.recv(consts.DEFAULT_BUFFER_SIZE))
        assert received_length == msg_length
        LOGGER.debug("finished send")
        time.sleep(consts.NET_DELAY)

    def rcv(self):
        LOGGER.debug("begin to receive")
        # receive msg length
        msg_length = pickle.loads(self._messenger.recv(consts.DEFAULT_BUFFER_SIZE))
        LOGGER.debug(f"received msg length: {msg_length}")
        time.sleep(consts.NET_DELAY)
        self._messenger.sendall(pickle.dumps(msg_length))
        time.sleep(consts.NET_DELAY)

        msg = self._messenger.recv(consts.DEFAULT_BUFFER_SIZE)
        time.sleep(consts.NET_DELAY)
        # receive msg from partner, the buffer size is defined in consts.py
        while len(msg) < msg_length:
            msg+=self._messenger.recv(consts.DEFAULT_BUFFER_SIZE)
            LOGGER.debug(f"received size: {len(msg)}")
        LOGGER.debug("received msg")
        time.sleep(consts.NET_DELAY)
        self._messenger.sendall(pickle.dumps(len(msg)))
        time.sleep(consts.NET_DELAY)
        return msg

    def add_nn_layer(self, layer):
        self._nn_model.add_module(name=f"layer {self._layer_index}", module=layer)
        self._layer_index += 1

        LOGGER.debug(f"add layer {layer} successfully")

    def set_optimizer(self, optimizer):
        old_optimizer = self._optimizer
        self._optimizer = optimizer
        LOGGER.debug(
            f"set optimizer successfully, optimizer changed: {old_optimizer} -> {self._optimizer}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nn_model(x)

    def backward(self, predicts: torch.Tensor, gradients_tensor: torch.Tensor):
        gradients_tensor = gradients_tensor.repeat(len(predicts),1)
        self._optimizer.zero_grad()
        predicts.backward(gradient=gradients_tensor)
        self._optimizer.step()

    def save_model(self):
        pickle.dump(self._nn_model,open(f"data/{self.m_param.role}_model","w"))

    def load_model(self):
        self._nn_model = pickle.load(open(f"data/{self.m_param.role}_model","r"))

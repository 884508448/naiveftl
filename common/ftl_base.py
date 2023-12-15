import socket
import utils.config as config
import torch
import pickle
import time
import numpy as np
import struct

from common.ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER
from typing import List


class FTLBase:
    def __init__(self, param: FTLParam):
        self.m_param = param
        self._nn_model = torch.nn.Sequential()
        self._layer_index = 0
        self._optimizer = None
        LOGGER.info(f"ftl {self.m_param.role} starting")

        LOGGER.debug("loading data")
        self.data_loader = FTLDataLoader(self.m_param.data_path)

        LOGGER.debug("building connection")
        self.__init_socket()

    def __init_socket(self):
        self.m_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.m_param.role == config.GUEST:
            self.m_sock.bind((config.DEFAULT_IP, config.GUEST_DEFAULT_PORT),)
            self.m_sock.listen()
            conn, addr = self.m_sock.accept()
            assert (
                addr == self.m_param.partner_addr
            ), f"get connection from: {addr}, not equal to offered: {self.m_param.partner_addr}"
            self._messenger: socket.socket = conn
        else:
            self.m_sock.bind((config.DEFAULT_IP, config.HOST_DEFAULT_PORT))
            self.m_sock.connect(self.m_param.partner_addr)
            self._messenger: socket.socket = self.m_sock
        LOGGER.debug("connection builded")

    def encrypt(self, matrix: np.array, public_key=None):
        if matrix is None:
            return None
        LOGGER.debug(f"encrypting ..., shape: {matrix.shape}")
        if public_key is None:
            public_key = self._public_key
        if matrix is None:
            return None
        en_matrix = np.array([public_key.encrypt(x)
                             for x in matrix.flatten().tolist()]).reshape(matrix.shape)
        return en_matrix

    def decrypt(self, en_matrix: np.array, private_key=None):
        if en_matrix is None:
            return None
        LOGGER.debug(f"decrypting ..., shape: {en_matrix.shape}")
        if private_key is None:
            private_key = self._private_key
        if en_matrix is None:
            return None
        de_matrix = np.array(
            [private_key.decrypt(x)
             for x in en_matrix.flatten().tolist()]
        ).reshape(en_matrix.shape)
        return de_matrix

    def add_partial_mask(self, partial):
        self._partial_mask = np.random.random(size=partial.shape)
        return partial + self._partial_mask

    def remove_partial_mask(self, masked_partial):
        return masked_partial - self._partial_mask

    def add_partial_mask_non_overlap(self, partial):
        self._partial_mask_non_overlap = np.random.random(size=partial.shape)
        return partial + self._partial_mask_non_overlap

    def remove_partial_mask_non_overlap(self, masked_partial):
        return masked_partial - self._partial_mask_non_overlap

    def display(self, name: str, obj):
        """
        debug util
        """
        if config.DEBUG:
            LOGGER.critical(f"{name}:\n{obj}")
            input()

    def send(self, msg: bytes):
        msg_length = struct.pack("!Q", len(msg))

        # send message length before the raw message to avoid sticky
        self._messenger.sendall(msg_length + msg)
        LOGGER.debug(f"send {len(msg)} bytes")

    def rcv(self):
        # receive 8 bytes message length
        length_data = self._messenger.recv(8)
        if length_data is None:
            LOGGER.error("received None length_data")
            return None

        msg_length = struct.unpack("!Q", length_data)[0]
        LOGGER.debug(f"received msg_length: {msg_length}")

        # receive message refer to msg_length
        data = b""
        while len(data) < msg_length:
            remaining = msg_length - len(data)
            received = self._messenger.recv(remaining)

            if not received:
                return None

            data += received

        LOGGER.debug(f"received {msg_length} bytes")
        return data

    def add_nn_layer(self, layer):
        self._nn_model.add_module(
            name=f"layer {self._layer_index}", module=layer)
        self._layer_index += 1

        LOGGER.debug(f"add layer {layer} successfully")

    def set_optimizer(self, optimizer=None):
        if optimizer is None:
            LOGGER.info(
                "optimizer has not been seted, it will be automatically seted as the default optimizer"
            )
            self.set_optimizer(
                optimizer=torch.optim.Adam(
                    self._nn_model.parameters(), lr=self.m_param.learning_rate)
            )
            return

        old_optimizer = self._optimizer
        self._optimizer = optimizer
        LOGGER.debug(
            f"set optimizer successfully, optimizer changed: {old_optimizer} -> {self._optimizer}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._nn_model(x)

    def backward(self, predicts: List[torch.Tensor], gradients_tensor: torch.Tensor):
        predicts = torch.stack(predicts)
        self._optimizer.zero_grad()
        predicts.backward(gradient=gradients_tensor)
        self._optimizer.step()

    def save_model(self):
        pickle.dump(self._nn_model, open(
            f"data/{self.m_param.role}_model", "w"))

    def load_model(self):
        self._nn_model = pickle.load(
            open(f"data/{self.m_param.role}_model", "r"))

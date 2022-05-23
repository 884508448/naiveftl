import socket
import utils.consts as consts

from ftl_param import FTLParam
from utils.ftl_data_loader import FTLDataLoader
from utils.ftl_log import LOGGER


class FTLBase:
    def __init__(self, param: FTLParam):
        self.m_param = param
        LOGGER.info(f"ftl {self.m_param.role} starting")

        LOGGER.debug("loading data")
        self.data_loader = FTLDataLoader(self.m_param.data_path)

        LOGGER.info("building connection")
        self.__init_socket()


    def __init_socket(self):
        self.m_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.m_param.role == consts.GUEST:
            self.m_sock.bind((consts.DEFAULT_IP, consts.GUEST_DEFAULT_PORT), )
            self.m_sock.listen()
            conn, addr = self.m_sock.accept()
            assert addr == self.m_param.partner_addr, f"get connection from: {addr}, not equal to offered: {self.m_param.partner_addr}"
            self.__messenger: socket.socket = conn
        else:
            self.m_sock.bind((consts.DEFAULT_IP, consts.HOST_DEFAULT_PORT))
            self.m_sock.connect(self.m_param.partner_addr)
            self.__messenger: socket.socket = self.m_sock

    def send(self, msg):
        self.__messenger.sendall(msg)

    def rcv(self):
        return self.__messenger.recv(consts.DEFAULT_BUFFER_SIZE)

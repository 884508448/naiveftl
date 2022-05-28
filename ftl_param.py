import os

from utils import consts


class FTLParam:
    def __init__(
        self,
        partner_addr,
        role,
        data_path,
        loss_tol=0.000001,
        const_k=-2,
        const_gamma=1,
        epochs=1,
        batch_size=-1,
        mode="plain",
    ):
        """
        Parameters
        ----------
        partner_addr: (ip:port)
            the ip:port address of the party
        role: str
            specifies whether the participant is host or guest
        loss_tol : float
            loss tolerance
        epochs : int
            epochs num
        batch_size : int
            batch size when computing transformed feature embedding, -1 use full data.
        mode: {"plain", "encrypted"}
            plain: will not use any encrypt algorithms, data exchanged in plaintext
            encrypted: use paillier to encrypt gradients
        """

        self.partner_addr = partner_addr
        self.role = role
        self.data_path = data_path
        self.loss_tol = loss_tol
        self.const_k = const_k
        self.const_gamma = const_gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.param_check()

    def param_check(self):
        if not isinstance(self.partner_addr, tuple):
            raise ValueError("partner_addr should be a tuple of (ip,port)")

        assert self.role in (
            "host",
            "guest",
        ), f"role options: host or guest, but {self.role} is offered"

        if not os.path.exists(self.data_path):
            raise ValueError(
                f"data_path does not exists! please check: {self.data_path}"
            )

        if not isinstance(self.loss_tol, (int, float)):
            raise ValueError("loss_tol should be numeric")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs should be a positive integer")

        if not isinstance(self.batch_size, int):
            raise ValueError("batch_size should be int")

        assert self.mode in (
            consts.ENCRYPTED_NODE,
            consts.PLAIN_MODE,
        ), f"mode options: encrypted or plain, but {self.mode} is offered"

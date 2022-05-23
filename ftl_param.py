
import os

from torch import nn

class FTLParam:
    def __init__(self, partner_addr, role, data_path, loss_tol=0.000001, const_k=-2, const_lambda=1,
                 nn_module=None, epochs=1, batch_size=-1,
                 mode='plain'):
        """
        Parameters
        ----------
        partner_addr: (ip:port)
            the ip:port address of the party
        role: str
            specifies whether the participant is host or guest
        loss_tol : float
            loss tolerance
        optimizer : str or dict
            optimizer method, accept following types:
            1. a string, one of "Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"
            2. a dict, with a required key-value pair keyed by "optimizer",
                with optional key-value pairs such as learning rate.
            defaults to "SGD"
        nn_module : torch.nn.Module
            a torch module defined by user
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
        self.const_lambda = const_lambda
        self.nn_module = nn_module
        self.epochs = epochs
        self.batch_size = batch_size
        self.mode = mode
        self.param_check()

    def param_check(self):
        if not isinstance(self.partner_addr, tuple):
            raise ValueError("partner_addr should be a tuple of (ip,port)")

        assert self.role in ("host", "guest"), f"role options: host or guest, but {self.role} is offered"

        if not os.path.exists(self.data_path):
            raise ValueError(f"data_path does not exists! please check: {self.data_path}")

        if not isinstance(self.loss_tol, (int, float)):
            raise ValueError("loss_tol should be numeric")

        if not isinstance(self.nn_module, nn.Module):
            raise ValueError("nn_module should be a module of pytorch")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs should be a positive integer")

        if not isinstance(self.batch_size, int):
            raise ValueError("batch_size should be int")

        assert self.mode in ("encrypted", "plain"), f"mode options: encrypted or plain, but {self.mode} is offered"

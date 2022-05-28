import torch
import torch.nn as nn

from ftl_host import FTLHost
from ftl_param import FTLParam
from utils import consts

h_p = {
    "partner_addr": (consts.DEFAULT_IP, consts.GUEST_DEFAULT_PORT),
    "role": "host",
    "data_path": "data/mini_nus_wide_train_host.csv",
}
host_param = FTLParam(**h_p)

host_ftl = FTLHost(host_param)
host_ftl.add_nn_layer(
    layer=nn.Linear(in_features=1000, out_features=32, dtype=torch.float32)
)
host_ftl.add_nn_layer(layer=nn.Sigmoid())
host_ftl.train()

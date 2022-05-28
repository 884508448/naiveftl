import torch
import torch.nn as nn

from ftl_guest import FTLGuest
from ftl_param import FTLParam
from utils import consts

g_p = {
    "partner_addr": (consts.DEFAULT_IP, consts.HOST_DEFAULT_PORT),
    "role": consts.GUEST,
    "data_path": "data/mini_nus_wide_train_guest.csv",
}
guest_param = FTLParam(**g_p)

guest_ftl = FTLGuest(guest_param)
guest_ftl.add_nn_layer(
    layer=nn.Linear(in_features=634, out_features=32, dtype=torch.float32)
)
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.train()

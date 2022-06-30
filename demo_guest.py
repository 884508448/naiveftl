import torch
import torch.nn as nn

from ftl_guest import FTLGuest
from ftl_param import FTLParam
from utils import consts

g_p = {
    "partner_addr": (consts.DEFAULT_IP, consts.HOST_DEFAULT_PORT),
    "role": consts.GUEST,
    # "data_path": "data/mini_nus_wide_train_guest.csv",
    "data_path": "data/nus_wide_train_guest.csv",
    # "data_path": "data/nus_wide_validate_guest.csv",
    "epochs":consts.EPOCHS,
    "const_k":consts.K,
    "const_gamma":consts.GAMMA,
    # "predict_data_path": "data/mini_nus_wide_validate_guest.csv",
    "predict_data_path": "data/nus_wide_validate_guest.csv",
    "learning_rate":0.01,
    "loss_tol":-100,
    "batch_size":consts.BATCH_SIZE
}
guest_param = FTLParam(**g_p)

guest_ftl = FTLGuest(guest_param)
guest_ftl.add_nn_layer(
    layer=nn.Linear(in_features=634, out_features=32, dtype=torch.float32)
)
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
# guest_ftl.add_nn_layer(layer=nn.Linear(in_features=32, out_features=32, dtype=torch.float32))
# guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.add_nn_layer(layer=nn.BatchNorm1d(num_features=32))


guest_ftl.train()
guest_ftl.predict()

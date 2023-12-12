import torch

from torch import nn
from ftl_guest import FTLGuest
from ftl_param import FTLParam
from utils import config

g_p = {
    "partner_addr": (config.DEFAULT_IP, config.HOST_DEFAULT_PORT),
    "role": config.GUEST,
    "epochs": config.EPOCHS,
    "const_k": config.K,
    "const_gamma": config.GAMMA,
    "learning_rate": config.LEARNING_RATE,
    "loss_tol": config.LOSS_TOLERANCE,
    "batch_size": config.BATCH_SIZE,
    "mode": config.MODE
}
if config.MINI_TEST:
    g_p.update({
        "data_path": config.MINI_GUEST_TRAIN_DATA_PATH,
        "predict_data_path": config.MINI_GUEST_VALIDATE_DATA_PATH,
    })
else:
    g_p.update({
        "data_path": config.GUEST_TRAIN_DATA_PATH,
        "predict_data_path": config.GUEST_VALIDATE_DATA_PATH,
    })
guest_param = FTLParam(**g_p)

guest_ftl = FTLGuest(guest_param)
guest_ftl.add_nn_layer(
    layer=nn.Linear(in_features=634, out_features=32, dtype=torch.float32)
)
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.add_nn_layer(layer=nn.Linear(
    in_features=32, out_features=32, dtype=torch.float32))
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.add_nn_layer(layer=nn.Linear(
    in_features=32, out_features=32, dtype=torch.float32))
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.add_nn_layer(layer=nn.Linear(
    in_features=32, out_features=32, dtype=torch.float32))
guest_ftl.add_nn_layer(layer=nn.Sigmoid())
guest_ftl.add_nn_layer(layer=nn.BatchNorm1d(num_features=32))

guest_ftl.train()

import torch

from torch import nn
from improved_ftl.ftl_guest import FTLGuest
from common.ftl_param import FTLParam
from utils import config


def train(id: int):
    g_p = {
        "partner_addr": (config.DEFAULT_IP, config.HOST_DEFAULT_PORT),
        "role": config.GUEST,
        "epochs": config.EPOCHS,
        "const_k": config.K,
        "const_gamma": config.GAMMA,
        "learning_rate": config.LEARNING_RATE,
        "loss_tol": config.LOSS_TOLERANCE,
        "batch_size": config.BATCH_SIZE,
        "mode": config.MODE,
        "predict_data_path": config.GUEST_VALIDATE_DATA_PATH,
        "data_path": f"data/nus_wide_train_guest_{id*50}.csv"
    }

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


if __name__ == "__main__":
    train(id=config.EXPERIMENT_ID)

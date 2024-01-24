import torch

from torch import nn
from improved_ftl.ftl_host import FTLHost
from common.ftl_param import FTLParam
from utils import config


def train(id: int):
    h_p = {
        "partner_addr": (config.DEFAULT_IP, config.GUEST_DEFAULT_PORT),
        "role": config.HOST,
        "epochs": config.EPOCHS,
        "const_k": config.K,
        "const_gamma": config.GAMMA,
        "learning_rate": config.LEARNING_RATE,
        "loss_tol": config.LOSS_TOLERANCE,
        "batch_size": config.BATCH_SIZE,
        "mode": config.MODE,
        "predict_data_path": config.HOST_VALIDATE_DATA_PATH,
        "data_path": f"data/nus_wide_train_host_{id*5}.csv"
    }

    host_param = FTLParam(**h_p)

    host_ftl = FTLHost(host_param)

    host_ftl.add_nn_layer(
        layer=nn.Linear(in_features=1000, out_features=config.MIDDLE_FEATURES, dtype=torch.float32)
    )
    host_ftl.add_nn_layer(layer=nn.Sigmoid())
    host_ftl.add_nn_layer(layer=nn.Linear(
        in_features=config.MIDDLE_FEATURES, out_features=config.MIDDLE_FEATURES, dtype=torch.float32))
    host_ftl.add_nn_layer(layer=nn.Sigmoid())
    host_ftl.add_nn_layer(layer=nn.Linear(
        in_features=config.MIDDLE_FEATURES, out_features=config.MIDDLE_FEATURES, dtype=torch.float32))
    host_ftl.add_nn_layer(layer=nn.Sigmoid())
    host_ftl.add_nn_layer(layer=nn.Linear(
        in_features=config.MIDDLE_FEATURES, out_features=config.MIDDLE_FEATURES, dtype=torch.float32))
    host_ftl.add_nn_layer(layer=nn.Sigmoid())
    host_ftl.add_nn_layer(layer=nn.BatchNorm1d(num_features=config.MIDDLE_FEATURES))
    host_ftl.train()


if __name__ == "__main__":
    train(id=config.EXPERIMENT_ID)

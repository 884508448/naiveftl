import torch
import torch.nn as nn

from ftl_host import FTLHost
from ftl_param import FTLParam
from utils import consts

h_p = {
    "partner_addr": (consts.DEFAULT_IP, consts.GUEST_DEFAULT_PORT),
    "role": "host",
    # "data_path": "data/mini_nus_wide_train_host.csv",
    "data_path": "data/nus_wide_train_host.csv",
    # "data_path": "data/nus_wide_validate_host.csv",
    "epochs":consts.EPOCHS,
    "const_k":consts.K,
    "const_gamma":consts.GAMMA,
    # "predict_data_path": "data/mini_nus_wide_validate_host.csv",
    "predict_data_path": "data/nus_wide_validate_host.csv",
    "learning_rate":0.01,
    "loss_tol":-100,
    "batch_size":consts.BATCH_SIZE
}
host_param = FTLParam(**h_p)

host_ftl = FTLHost(host_param)
host_ftl.add_nn_layer(
    layer=nn.Linear(in_features=1000, out_features=32, dtype=torch.float32)
)
host_ftl.add_nn_layer(layer=nn.Sigmoid())
# host_ftl.add_nn_layer(layer=nn.Linear(in_features=32, out_features=32, dtype=torch.float32))
# host_ftl.add_nn_layer(layer=nn.Sigmoid())
host_ftl.add_nn_layer(layer=nn.BatchNorm1d(num_features=32))
host_ftl.train()

results,accuracy=host_ftl.predict()
print(f"results: {results}\naccuracy: {accuracy}")
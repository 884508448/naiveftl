import torch
import torch.nn as nn
import numpy as np

from utils.ftl_data_loader import FTLDataLoader

# dt = FTLDataLoader("data/mini_nus_wide_train_host.csv")


# model = nn.Sequential()
# model.add_module(
#     "layer 1", nn.Linear(in_features=1000, out_features=32, dtype=torch.float32)
# )
# model.add_module("layer 1 activation", nn.Sigmoid())

# # y = np.ndarray()
# # out = model(torch.tensor(y, dtype=torch.float32))
# # y = np.concatenate(y, dt.labels[0:3])
# # print(np.dot(y, np.ones_like(y)))
# print(type(torch.optim.Adam(model.parameters(), lr=0.001)))
# print(f"data matrix: {dt.data_matrix}")
# print(f"0row: {dt.data_matrix[0]}")
# out = model(torch.tensor(dt.data_matrix[0], dtype=torch.float32)).detach().numpy()
# print(f"out: {out}")
# print(f"out*out.T: {np.dot(out,out)}")
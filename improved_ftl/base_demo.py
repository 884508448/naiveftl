import torch.nn as nn
import numpy as np

from phe import paillier
from time import time
from functools import wraps

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

def timer(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        start = time()
        r = func()
        time_cost = time()-start
        print(f"time cost: {time_cost}")
        return r
    return decorated


@timer
def test():
    public_key, private_key = paillier.generate_paillier_keypair()

    for i in range(50):
        x = 123.56
        y = 456.78

        x = public_key.encrypt(x)
        y = public_key.encrypt(y)

        z = x+y


print(test())

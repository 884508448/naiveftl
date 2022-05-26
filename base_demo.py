import torch
import torch.nn as nn

from utils.ftl_data_loader import FTLDataLoader

dt = FTLDataLoader("data/mini_nus_wide_train_guest.csv")

y=dt
model = nn.Sequential()
model.add_module("layer 1",nn.Linear(in_features=634,out_features=32,dtype=torch.float32))
model.add_module("layer 1 activation",nn.Sigmoid())

out=model(torch.tensor(y,dtype=torch.float32))
y=dt.labels

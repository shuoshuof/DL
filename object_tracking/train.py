import math
import time
import numpy as np
import torch
import skimage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional
# class MLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(42, 256)  # 隐藏层
#         self.out = nn.Linear(256, 3)  # 输出层
#     def forward(self, X):
#         X= functional.relu(self.hidden(X))
#         X=self.out(X)
#         X=functional.softmax(X, dim = 1)
#         return X
net =nn.Sequential(nn.Flatten(),
                   nn.Linear(42,256),
                   nn.ReLU(),
                   nn.Linear(256,3))
# net = nn.Sequential(nn.Flatten(),
#                     nn.Linear(784, 256),
#                     nn.ReLU(),
#                     nn.Linear(256, 10))

print(net)
params = list(net.parameters())
print(len(params))
for param in params:
    print(param.size())
    # print(param.__class__.__name__)
X = torch.rand(size=(1,42), dtype=torch.float32)
print(net(X))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape: \t',X.shape)

loss = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器


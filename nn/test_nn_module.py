from typing import TypeVar

import torch
import torch.nn.functional as F
from torch import nn, Tensor

T = TypeVar('T')

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

    # 正向传播
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

my_module = MyModule()
x = torch.randn(5, 3, 64, 64)

y = my_module(x)
print(y)
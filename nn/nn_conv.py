import torch
import torch.nn.functional as F

input = torch.tensor(
    [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]
    ]
)

kernel = torch.tensor(
    [
        [1,2,1],
        [0,1,0],
        [2,1,0],
    ]
)

# 在第0维位置插入一个大小为1的新维度
# (5, 5) -> (1, 5, 5) -> (1, 1, 5, 5)
input_4d = input.unsqueeze(0).unsqueeze(0)

kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

# stride
output = F.conv2d(input_4d, kernel_4d, stride=1)
print(output)

output1 = F.conv2d(input_4d, kernel_4d, stride=2)
print(output1)

# 填充值为0
output2 = F.conv2d(input_4d, kernel_4d, stride=1, padding=1)
print(output2)

import torch
from torch_spline_conv import spline_conv

x = torch.Tensor([[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
index = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
pseudo = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
pseudo = torch.Tensor(pseudo)
weight = torch.arange(0.5, 0.5 * 25, step=0.5).view(12, 2, 1)
# print(weight[:, 0].squeeze())
kernel_size = torch.LongTensor([3, 4])
is_open_spline = torch.ByteTensor([1, 0])
root_weight = torch.arange(12.5, 13.5, step=0.5).view(2, 1)

output = spline_conv(x, index, pseudo, weight, kernel_size, is_open_spline,
                     root_weight)

edgewise_output = [
    1 * 0.25 * (0.5 + 1.5 + 4.5 + 5.5) + 2 * 0.25 * (1 + 2 + 5 + 6),
    3 * 0.25 * (1.5 + 2.5 + 5.5 + 6.5) + 4 * 0.25 * (2 + 3 + 6 + 7),
    5 * 0.25 * (6.5 + 7.5 + 10.5 + 11.5) + 6 * 0.25 * (7 + 8 + 11 + 12),
    7 * 0.25 * (7.5 + 4.5 + 11.5 + 8.5) + 8 * 0.25 * (8 + 5 + 12 + 9),
]

expected_output = [
    [12.5 * 9 + 13 * 10 + sum(edgewise_output) / 4],
    [12.5 * 1 + 13 * 2],
    [12.5 * 3 + 13 * 4],
    [12.5 * 5 + 13 * 6],
    [12.5 * 7 + 13 * 8],
]

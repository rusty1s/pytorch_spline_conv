import time

import torch
from torch_spline_conv.functions.ffi import spline_basis_forward

n, d = 9999999, 5
pseudo = torch.FloatTensor(n, d).uniform_(0, 1)
kernel_size = torch.LongTensor(d).fill_(5)
is_open_spline = torch.ByteTensor(d).fill_(1)
K = kernel_size.prod()

t = time.perf_counter()
basis, index = spline_basis_forward(1, pseudo, kernel_size, is_open_spline, K)
t = time.perf_counter() - t
print('CPU:', t)

pseudo = pseudo.cuda()
kernel_size = kernel_size.cuda()
is_open_spline = is_open_spline.cuda()

torch.cuda.synchronize()
t = time.perf_counter()
basis, index = spline_basis_forward(1, pseudo, kernel_size, is_open_spline, K)
torch.cuda.synchronize()
t = time.perf_counter() - t
print('GPU:', t)

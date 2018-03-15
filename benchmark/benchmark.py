import time
import sys

import torch
from torch.autograd import Variable
from torch_spline_conv import spline_conv

sys.path.insert(0, '../../pytorch_geometric')

from torch_geometric.nn.modules import SplineConv  # noqa

n = 600
m_in = 64
m_out = 64
d = 4
x = torch.FloatTensor(n, m_in).uniform_(-1, 1)
row = torch.arange(0, n).view(-1, 1).repeat(1, n).view(-1).long()
col = torch.arange(0, n).repeat(n).long()
edge_index = torch.stack([row, col], dim=0)
pseudo = torch.FloatTensor(n * n, d).uniform_(0, 1)
kernel_size = torch.LongTensor(d).fill_(5)
is_open_spline = torch.ByteTensor(d).fill_(1)
K = kernel_size.prod()
weight = torch.FloatTensor(K, m_in, m_out).uniform_(-1, 1)

# t = time.perf_counter()
# out = spline_conv(x, edge_index, pseudo, weight, kernel_size, is_open_spline)
# t = time.perf_counter() - t
# print('CPU:', t)

x = x.cuda()
edge_index = edge_index.cuda()
pseudo = pseudo.cuda()
weight = weight.cuda()
kernel_size = kernel_size.cuda()
is_open_spline = is_open_spline.cuda()

out = spline_conv(x, edge_index, pseudo, weight, kernel_size, is_open_spline)
torch.cuda.synchronize()
t = time.perf_counter()
out = spline_conv(x, edge_index, pseudo, weight, kernel_size, is_open_spline)
torch.cuda.synchronize()
t = time.perf_counter() - t
print('GPU:', t)

conv = SplineConv(m_in, m_out, d, kernel_size, is_open_spline.long()).cuda()
adj = {'indices': edge_index, 'values': pseudo, 'size': torch.Size([n, n, d])}
x = Variable(x)
conv(adj, x)
torch.cuda.synchronize()
t = time.perf_counter()
conv(adj, x)
torch.cuda.synchronize()
t = time.perf_counter() - t
print('GPU old:', t)

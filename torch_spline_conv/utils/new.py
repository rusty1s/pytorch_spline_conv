import torch
from torch.autograd import Variable


def new(x, *sizes):
    return x.new(sizes) if torch.is_tensor(x) else Variable(x.data.new(sizes))

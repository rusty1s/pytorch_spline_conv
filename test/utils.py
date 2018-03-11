import torch

tensors = ['FloatTensor', 'DoubleTensor']


def Tensor(str, x):
    tensor = getattr(torch, str)
    return tensor(x)

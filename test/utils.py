import torch

tensors = ['FloatTensor']


def Tensor(str, x):
    tensor = getattr(torch, str)
    return tensor(x)

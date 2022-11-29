from typing import Any

import torch

dtypes = [torch.float, torch.double, torch.bfloat16]

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda:0')]


def tensor(x: Any, dtype: torch.dtype, device: torch.device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)

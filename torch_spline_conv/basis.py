from typing import Tuple

import torch


@torch.jit.script
def spline_basis(pseudo: torch.Tensor, kernel_size: torch.Tensor,
                 is_open_spline: torch.Tensor,
                 degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_spline_conv.spline_basis(pseudo, kernel_size,
                                                    is_open_spline, degree)

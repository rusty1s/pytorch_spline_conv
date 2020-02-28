import torch


@torch.jit.script
def spline_weighting(x: torch.Tensor, weight: torch.Tensor,
                     basis: torch.Tensor,
                     weight_index: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv.spline_weighting(
        x, weight, basis, weight_index)

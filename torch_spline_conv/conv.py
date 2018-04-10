# import torch


def spline_conv(x,
                edge_index,
                pseudo,
                weight,
                kernel_size,
                is_open_spline,
                degree=1,
                root_weight=None,
                bias=None):
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

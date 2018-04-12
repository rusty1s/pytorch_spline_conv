[pypi-image]: https://badge.fury.io/py/torch-spline-conv.svg
[pypi-url]: https://pypi.python.org/pypi/torch-spline-conv
[build-image]: https://travis-ci.org/rusty1s/pytorch_spline_conv.svg?branch=master
[build-url]: https://travis-ci.org/rusty1s/pytorch_spline_conv
[coverage-image]: https://codecov.io/gh/rusty1s/pytorch_spline_conv/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/rusty1s/pytorch_spline_conv?branch=master

# Spline-Based Convolution Operator of SplineCNN

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]

--------------------------------------------------------------------------------

This is a PyTorch implementation of the spline-based convolution operator of SplineCNN, as described in our paper:

Matthias Fey, Jan Eric Lenssen, Frank Weichert, Heinrich Müller: [SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels](https://arxiv.org/abs/1711.08920) (CVPR 2018)

The operator works on all floating point data types and is implemented both for CPU and GPU.

## Installation

If cuda is available, check that `nvcc` is accessible from your terminal, e.g. by typing `nvcc --version`.
If not, add cuda (`/usr/local/cuda/bin`) to your `$PATH`.
Then run:

```
pip install cffi torch-spline-conv
```

## Usage

```python
from torch_spline_conv import spline_conv

output = spline_conv(src, edge_index, pseudo, weight, kernel_size,
                     is_open_spline, degree=1, root_weight=None, bias=None)
```

Applies the spline-based convolutional operator
<p align="center">
  <img width="50%" src="https://user-images.githubusercontent.com/6945922/38684093-36d9c52e-3e6f-11e8-9021-db054223c6b9.png" />
</p>
over several node features of an input graph.
The kernel function g is defined over the weighted B-spline tensor product basis, as shown below for different B-spline degrees.

<p align="center">
  <img width="45%" src="https://user-images.githubusercontent.com/6945922/38685443-3a2a0c68-3e72-11e8-8e13-9ce9ad8fe43e.png" />
  <img width="45%" src="https://user-images.githubusercontent.com/6945922/38685459-42b2bcae-3e72-11e8-88cc-4b61e41dbd93.png" />
</p>

### Parameters

* **src** *(Tensor or Variable)* - Input node features of shape `(number_of_nodes x in_channels)`
* **edge_index** *(LongTensor)* - Graph edges, given by source and target indices, of shape `(2 x number_of_edges)`
* **pseudo** *(Tensor or Variable)* - Edge attributes, ie. pseudo coordinates, of shape `(number_of_edges x number_of_edge_attributes)` in the fixed interval [0, 1]
* **weight** *(Tensor or Variable)* - Trainable weight parameters of shape `(kernel_size x in_channels x out_channels)`
* **kernel_size** *(LongTensor)* - Number of trainable weight parameters in each edge dimension
* **is_open_spline** *(ByteTensor)* - Whether to use open or closed B-spline bases for each dimension
* **degree** *(int)* - B-spline basis degree (default: `1`)
* **root_weight** *(Tensor or Variable)* - Additional shared trainable parameters for each feature of the root node of shape `(in_channels x out_channels)` (default: `None`)
* **bias** *(Tensor or Variable)* - Optional bias of shape (out_channels) (default: `None`)

### Example

```python
import torch
from torch_spline_conv import spline_conv

src = torch.Tensor(4, 2)  # 4 nodes with 2 features each
edge_index = torch.LongTensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])  # 6 edges
pseudo = torch.Tensor(6, 2)  # two-dimensional edge attributes
weight = torch.Tensor(25, 2, 4)  # 25 trainable parameters for each in_channels x out_channels combination
kernel_size = torch.LongTensor([5, 5])  # 5 trainable parameters in each edge dimension
is_open_spline = torch.ByteTensor([1, 1])  # only use open B-splines
degree = 1  # B-spline degree of 1
root_weight = torch.Tensor(2, 4)  # Weight root nodes separately
bias = None  # No additional bias

output = spline_conv(src, edge_index, pseudo, weight, kernel_size,
                     is_open_spline, degree, root_weight, bias)

print(output.size())
torch.Size([4, 4])  # 4 nodes with 4 features each
```

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{Fey/etal/2018,
  title={{SplineCNN}: Fast Geometric Deep Learning with Continuous {B}-Spline Kernels},
  author={Fey, Matthias and Lenssen, Jan Eric and Weichert, Frank and M{\"u}ller, Heinrich},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
  year={2018},
}
```

## Running tests

```
python setup.py test
```

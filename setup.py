from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

extra_compile_args = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    extra_compile_args += ['-DVERSION_GE_1_3']

ext_modules = [
    CppExtension('torch_spline_conv.basis_cpu', ['cpu/basis.cpp'],
                 extra_compile_args=extra_compile_args),
    CppExtension('torch_spline_conv.weighting_cpu', ['cpu/weighting.cpp'],
                 extra_compile_args=extra_compile_args),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}

if CUDA_HOME is not None:
    ext_modules += [
        CUDAExtension('torch_spline_conv.basis_cuda',
                      ['cuda/basis.cpp', 'cuda/basis_kernel.cu'],
                      extra_compile_args=extra_compile_args),
        CUDAExtension('torch_spline_conv.weighting_cuda',
                      ['cuda/weighting.cpp', 'cuda/weighting_kernel.cu'],
                      extra_compile_args=extra_compile_args),
    ]

__version__ = '1.1.1'
url = 'https://github.com/rusty1s/pytorch_spline_conv'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_spline_conv',
    version=__version__,
    description=('Implementation of the Spline-Based Convolution Operator of '
                 'SplineCNN in PyTorch'),
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'spline-cnn',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
)

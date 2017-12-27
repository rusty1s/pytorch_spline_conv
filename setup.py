from os import path as osp
from setuptools import setup, find_packages

import build  # noqa

install_requires = ['cffi']
setup_requires = ['pytest-runner', 'cffi']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_spline_conv',
    version='0.1.0',
    description='PyTorch extension for spline-based convolutions',
    url='https://github.com/rusty1s/pytorch_spline_conv',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[osp.join(osp.dirname(__file__), 'build.py:ffi')],
)

import sys
import os
import shutil
import subprocess

import torch
from torch.utils.ffi import create_extension

if os.path.exists('build'):
    shutil.rmtree('build')

headers = ['torch_spline_conv/src/cpu.h']
sources = ['torch_spline_conv/src/cpu.c']
include_dirs = ['torch_spline_conv/src']
define_macros = []
extra_objects = []
with_cuda = False

if torch.cuda.is_available():
    subprocess.call('./build.sh {}'.format(sys.executable))

    headers += ['torch_spline_conv/src/cuda.h']
    sources += ['torch_spline_conv/src/cuda.c']
    include_dirs += ['torch_spline_conv/kernel']
    define_macros += [('WITH_CUDA', None)]
    extra_objects += ['torch_spline_conv/build/kernel.so']
    with_cuda = True

ffi = create_extension(
    name='torch_spline_conv._ext.ffi',
    package=True,
    headers=headers,
    sources=sources,
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_objects=extra_objects,
    with_cuda=with_cuda,
    relative_to=__file__)

if __name__ == '__main__':
    ffi.build()

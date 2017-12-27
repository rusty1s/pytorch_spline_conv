from torch.utils.ffi import create_extension

headers = ['torch_spline_conv/src/cpu.h']
sources = ['torch_spline_conv/src/cpu.c']
include_dirs = ['torch_spline_conv/src']
define_macros = []
extra_objects = []
with_cuda = False

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

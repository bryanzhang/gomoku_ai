#! /usr/bin/python3

import os
from setuptools import setup, Extension
import pybind11

# NOTE(junhaozhang): distutils ignores Extension(language='clang++') and
# defaults to g++. The hand-written pointer packing in this project (the
# high 16 bits of children_ store the move index) is miscompiled by g++ -O3
# strict-aliasing optimizations (RolloutWithModel segfaults at runtime), so
# clang++ must be used -- force it as the default here.
os.environ.setdefault('CC', 'clang')
os.environ.setdefault('CXX', 'clang++')

import torch  # used to locate libtorch headers and shared libraries

torch_dir = os.path.dirname(torch.__file__)
torch_include = os.path.join(torch_dir, 'include')
torch_lib = os.path.join(torch_dir, 'lib')

ext_modules = [
    Extension(
        'gomoku_ai',
        ['alphazero_mcts.cpp'],
        include_dirs=[pybind11.get_include(), torch_include],
        libraries=['stdc++',],
        library_dirs=['/usr/local/lib', torch_lib],
        language='clang++',
        #language='g++',
        #extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC', '-fsanitize=address', '-fno-omit-frame-pointer'],
        extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC',],
        # NOTE(junhaozhang): the NEEDED entries of the torch libraries must
        # be kept -- Debian's linker defaults to --as-needed and would drop
        # them, leaving symbols such as the AutogradMeta vtable in the .so
        # undefined at load time (defined in libtorch_cpu, pulled into the
        # global scope by libtorch_cuda which is loaded with RTLD_GLOBAL on
        # `import torch`).
        # NOTE(junhaozhang): -lgomp resolves omp_set_num_threads in
        # RolloutWithModel (disables OMP parallelism of oneDNN conv; see the
        # comments in the hpp); at runtime it reuses the libgomp already
        # loaded by torch.
        extra_link_args=['-g', '-Wl,--no-as-needed', '-L/usr/local/lib', '-lfolly', '-ldl', '-lgflags', '-lglog', '-lpthread', '-lfmt', '-lunwind', '-ldouble-conversion', '-liberty', '-lstdc++', '-levent', '-lboost_context', '-lgomp', f'-L{torch_lib}', '-ltorch_cuda', '-lc10_cuda', '-ltorch_global_deps', '-ltorch', '-lc10', f'-Wl,-rpath,{torch_lib}'],
    ),
]

setup(
    name='gomoku_ai',
    version='0.1.0',
    ext_modules=ext_modules,
)

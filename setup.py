#! /usr/bin/python3

import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gomoku_ai',
        ['alphazero_mcts.cpp'],
        include_dirs=[pybind11.get_include(), '/usr/local/lib/python3.9/dist-packages/torch/include/'],
        libraries=['stdc++',],
        library_dirs=['/usr/local/lib', '/usr/local/lib/python3.9/dist-packages/torch/lib'],
        language='clang++',
        #language='g++',
        #extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC', '-fsanitize=address', '-fno-omit-frame-pointer'],
        extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC',],
        extra_link_args=['-g', '-L/usr/local/lib', '-lfolly', '-ldl', '-lgflags', '-lglog', '-lpthread', '-lfmt', '-lunwind', '-ldouble-conversion', '-liberty', '-lstdc++', '-levent', '-lboost_context', '-L/usr/local/lib/python3.9/dist-packages/torch/lib', '-ltorch_cuda', '-lc10_cuda', '-ltorch_global_deps', '-ltorch', '-lc10',],
    ),
]

setup(
    name='gomoku_ai',
    version='0.1.0',
    ext_modules=ext_modules,
)

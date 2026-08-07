#! /usr/bin/python3

import os
from setuptools import setup, Extension
import pybind11

# NOTE(junhaozhang): distutils 会忽略 Extension(language='clang++') 而默认用 g++。
# 本工程的手写指针打包代码(children_ 高 16 位存 idx)在 g++ -O3 严格别名优化下会被
# 误编译(运行时 RolloutWithModel 段错误), 必须用 clang++ 构建, 故在此强制默认。
os.environ.setdefault('CC', 'clang')
os.environ.setdefault('CXX', 'clang++')

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
        # NOTE(junhaozhang): 必须保留 torch 库的 NEEDED 项——Debian 链接器默认 --as-needed
        # 会把它们丢掉, 导致 .so 里的 AutogradMeta vtable 等符号在加载时找不到定义
        # (定义在 libtorch_cpu, 由 import torch 时 RTLD_GLOBAL 加载的 libtorch_cuda 带入全局)。
        extra_link_args=['-g', '-Wl,--no-as-needed', '-L/usr/local/lib', '-lfolly', '-ldl', '-lgflags', '-lglog', '-lpthread', '-lfmt', '-lunwind', '-ldouble-conversion', '-liberty', '-lstdc++', '-levent', '-lboost_context', '-L/usr/local/lib/python3.9/dist-packages/torch/lib', '-ltorch_cuda', '-lc10_cuda', '-ltorch_global_deps', '-ltorch', '-lc10',],
    ),
]

setup(
    name='gomoku_ai',
    version='0.1.0',
    ext_modules=ext_modules,
)

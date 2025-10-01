#! /usr/bin/python3

import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gomoku_ai',
        ['gomoku_ai.cpp'],
        include_dirs=[pybind11.get_include()],
        libraries=['stdc++',],
        library_dirs=['/usr/local/lib'],
        language='clang++',
        #language='g++',
        #extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC', '-fsanitize=address', '-fno-omit-frame-pointer'],
        extra_compile_args=['-x', 'c++', '-std=c++17', '-g', '-O3', '-fPIC',],
        extra_link_args=['-L/usr/local/lib', '-lfolly', '-ldl', '-lgflags', '-lglog', '-lpthread', '-lfmt', '-lunwind', '-ldouble-conversion', '-liberty', '-lstdc++', '-levent', '-lboost_context',],
    ),
]

setup(
    name='gomoku_ai',
    version='0.1.0',
    ext_modules=ext_modules,
)

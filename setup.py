#! /usr/bin/python3

import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'gomoku_ai',
        ['gomoku_ai.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++20', '-g', '-O3'],
    ),
]

setup(
    name='gomoku_ai',
    version='0.1.0',
    ext_modules=ext_modules,
)

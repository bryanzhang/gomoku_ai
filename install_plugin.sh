#! /bin/bash
# Build the gomoku_ai C++ extension in place (produces gomoku_ai.*.so).
set -e

export CC=${CC:-clang}
export CXX=${CXX:-clang++}
#export CC=gcc
#export CXX=g++

rm -rf gomoku_ai.*.so build
python3 setup.py clean
python3 setup.py build_ext --inplace --verbose

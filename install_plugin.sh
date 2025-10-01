#! /bin/bash

export CC=clang
export CXX=clang++
#export CC=gcc
#export CXX=g++
rm -rf gomoku_ai.cpython-39-x86_64-linux-gnu.so build
./setup.py clean
./setup.py build_ext --inplace --verbose

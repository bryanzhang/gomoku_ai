#! /bin/bash
# Build the C++ web server (Crow + embedded Python + libtorch).
set -e

CXX=${CXX:-clang++}

# Auto-detect torch / pybind11 / python paths from the current interpreter.
TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
PYBIND11_INC=$(python3 -c 'import pybind11; print(pybind11.get_include())')
PYTHON_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PYTHON_LIB=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PYTHON_VER=$(python3 -c 'import sysconfig; print(sysconfig.get_python_version())')

# NOTE(junhaozhang): -lgomp resolves omp_set_num_threads in
# alphazero_mcts.hpp (worker threads disable OMP parallelism of conv); at
# runtime it reuses the libgomp already loaded by torch.
$CXX web_server.cpp -o web_server -g \
    -I"$TORCH_DIR/include" -I"$PYBIND11_INC" -I"$PYTHON_INC" \
    -I./third_party/Crow/include -I./third_party/json/include \
    -std=c++17 -O3 -fPIC \
    -L/usr/local/lib -lfolly -ldl -lgflags -lglog -lpthread -lfmt -lunwind \
    -ldouble-conversion -liberty -lstdc++ -levent -lboost_context \
    -L"$PYTHON_LIB" -lpython"$PYTHON_VER" -latomic -ljemalloc -lgomp \
    -L"$TORCH_DIR/lib" -ltorch_cuda -ltorch -ltorch_cpu -lc10 -lc10_cuda \
    -Wl,-rpath,"$TORCH_DIR/lib"

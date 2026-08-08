#! /bin/bash

# NOTE(junhaozhang): -lgomp 解析 alphazero_mcts.hpp 里的 omp_set_num_threads
# (worker 线程关闭 conv 的 OMP 并行); 运行时复用 torch 已加载的 libgomp。
clang++ web_server.cpp -o web_server -g -I/usr/local/lib/python3.9/dist-packages/torch/include/ -I/usr/local/lib/python3.9/dist-packages/pybind11/include -I/usr/include/python3.9 -I./third_party/Crow/include -I./third_party/json/include -std=c++17 -O3 -fPIC -L/usr/local/lib -lfolly -ldl -lgflags -lglog -lpthread -lfmt -lunwind -ldouble-conversion -liberty -lstdc++ -levent -lboost_context -L/usr/lib -lpython3.9 -latomic -ljemalloc -lgomp -L/usr/local/lib/python3.9/dist-packages/torch/lib -ltorch_cuda -ltorch -ltorch_cpu -lc10 -lc10_cuda -Wl,-rpath,/usr/local/lib/python3.9/dist-packages/torch/lib

#! /bin/bash

clang++ web_server.cpp -g -I/usr/local/lib/python3.9/dist-packages/pybind11/include -I/usr/include/python3.9 -I./third_party/Crow/include -I./third_party/json/include -std=c++17 -O3 -fPIC -L/usr/local/lib -lfolly -ldl -lgflags -lglog -lpthread -lfmt -lunwind -ldouble-conversion -liberty -lstdc++ -levent -lboost_context -L/usr/lib

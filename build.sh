#!/bin/bash
set -ex

g++ -O3 -std=c++17 avs_example.cpp -ldnnl -lfaiss -fopenmp -lopenblas -march=native -o avs_example

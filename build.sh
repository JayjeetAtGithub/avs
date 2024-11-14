#!/bin/bash
set -ex

g++ -O3 -std=c++17 avs_example.cpp -ldnnl -march=native -o avs_example
g++ -O3 -std=c++17 avs_example_perf.cpp -ldnnl -march=native -o avs_example_perf

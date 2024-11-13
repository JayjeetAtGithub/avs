#!/bin/bash
set -ex

g++ -O3 -std=c++17 example.cpp -ldnnl -march=native -o avs_example


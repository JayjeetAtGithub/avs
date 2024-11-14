# Accelerated Vector Search

In this project, we are utilizing the Intel AMX (Advanced Matrix Extensions) accelerator accelerate
Inner Product and Euclidean distance calculations in vector search algorithms. This project is very much a WIP at this point. We have a very basic implementation done so far and we currently accelerate
only brute force searches. Please follow the instructions below to use and contribute to our project.

## Building from source

1. Install the oneDNN library.

```bash
git clone https://github.com/oneapi-src/oneDNN
cd oneDNN/
mkdir build/
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release ..
sudo make -j$(nproc) install
```

2. Build this project.
```bash
./build.sh
```

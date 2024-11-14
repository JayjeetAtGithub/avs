# Accelerated Vector Search

In this project, we are utilizing the Intel AMX (Advanced Matrix Extensions) accelerator accelerate
Inner Product and Euclidean distance calculations in vector search algorithms. This project is very much a WIP at this point. We have a very basic implementation done so far and we currently accelerate
only brute force searches. Please follow the instructions below to use and contribute to our project.

## Building from source

**Note:** An Intel processor from the Sapphire Rapids lineup is required for this project.

1. Install the oneDNN library.

```bash
git clone https://github.com/oneapi-src/oneDNN
cd oneDNN/
mkdir build/
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release ..
sudo make -j$(nproc) install
```

2. Build the project.
```bash
./build.sh
```

3. Run the example with verbose logging.
```bash
export LD_LIBRARY_PATH=/usr/local/lib
export ONEDNN_VERBOSE=1
./avs_example -d 16 -k 10 -b 2048 --nd 8192 --nq 4096
```

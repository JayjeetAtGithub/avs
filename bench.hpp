#pragma once

#include <chrono>
#include <random>
#include "dist.hpp"

namespace avs {

class Benchmark {
public:
    bool only_amx = false;
    dnnl::engine engine;
    dnnl::stream stream;

    void run_ip_1_x_N(uint64_t size) {
        uint64_t mat_a_size = 1;
        uint64_t mat_a_dim = size;
        uint64_t mat_b_size = size;
        uint64_t mat_b_dim = size;

        std::vector<float> mat_a(mat_a_size * mat_a_dim);
        std::vector<float> mat_b(mat_b_size * mat_b_dim);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<float> distrib;

        for (int i = 0; i < mat_a_size; i++) {
            for (int j = 0; j < mat_a_dim; j++) {
                mat_a[i * mat_a_dim + j] = distrib(rng);
            }
        }
        for (int i = 0; i < mat_b_size; i++) {
            for (int j = 0; j < mat_b_dim; j++) {
                mat_b[i * mat_b_dim + j] = distrib(rng);
            }
        }

        std::cout << "Data size: " << ((double)(size * size * 4) + (double)(size * 1 * 4)) / pow(10, 6) << " MiB" << std::endl;
        uint64_t total_flop = (uint64_t)mat_b_size * (2 * (uint64_t)mat_a_dim - 1);
        std::cout << "FLOP: " << total_flop << std::endl;

        if (!only_amx) {
            auto start = std::chrono::high_resolution_clock::now();
            ip_distance_avx512(mat_a.data(), mat_b.data(), mat_b_size, mat_b_dim, engine, stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "IP 1 x N AVX512: " << dur << " ms" << std::endl;
            std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            ip_distance_amx(mat_a.data(), mat_b.data(), mat_a_size, mat_b_size, mat_b_dim, engine, stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "IP 1 x N AMX: " << dur << " ms" << std::endl;
            std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
        }
    }

    void run_ip_N_x_N(uint64_t size) {
        uint64_t mat_a_size = size;
        uint64_t mat_a_dim = size;
        uint64_t mat_b_size = size;
        uint64_t mat_b_dim = size;

        std::vector<float> mat_a(mat_a_size * mat_a_dim);
        std::vector<float> mat_b(mat_b_size * mat_b_dim);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<float> distrib;

        for (int i = 0; i < mat_a_size; i++) {
            for (int j = 0; j < mat_a_dim; j++) {
                mat_a[i * mat_a_dim + j] = distrib(rng);
            }
        }
        for (int i = 0; i < mat_b_size; i++) {
            for (int j = 0; j < mat_b_dim; j++) {
                mat_b[i * mat_b_dim + j] = distrib(rng);
            }
        }

        std::cout << "Data size: " << (double)(size * size * 8) / pow(10, 6) << " MiB" << std::endl;

        _mm_prefetch(mat_a.data(), _MM_HINT_T2);
        _mm_prefetch(mat_b.data(), _MM_HINT_T2);

        uint64_t total_flop = (uint64_t)mat_a_size * (uint64_t)mat_b_size * (2 * (uint64_t)mat_a_dim - 1);
        std::cout << "FLOP: " << total_flop << std::endl;

        if (!only_amx) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < mat_a_size; i++) {
                ip_distance_avx512(mat_a.data() + i * mat_a_dim, mat_b.data(), mat_b_size, mat_b_dim, engine, stream);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "IP N x N AVX512: " << dur << " ms" << std::endl;
            std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            ip_distance_amx(mat_a.data(), mat_b.data(), mat_a_size, mat_b_size, mat_b_dim, engine, stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "IP N x N AMX: " << dur << " ms" << std::endl;
            std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
        }
    }
};

void run_bench() {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);
    
    Benchmark bench;
    bench.engine = engine;
    bench.stream = stream;

    // Just bench AMX
    bench.only_amx = true;
    std::vector<uint64_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    for (auto size : sizes) {
        bench.run_ip_N_x_N(size);
    }

    // Compare AMX and AVX512
    bench.only_amx = false;
    bench.run_ip_1_x_N(8192);
    bench.run_ip_N_x_N(8192);
}

} // namespace avs

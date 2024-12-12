#pragma once

#include <chrono>
#include <random>
#include "dist.hpp"

#define CONST_A 8192

namespace avs {

void run_ip_1_x_N(dnnl::engine &engine, dnnl::stream &stream) {
    int32_t mat_a_size = 1;
    int32_t mat_a_dim = CONST_A;
    int32_t mat_b_size = CONST_A;
    int32_t mat_b_dim = CONST_A;

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

    int64_t total_flop = (int64_t)mat_b_size * (2 * (int64_t)mat_a_dim - 1);
    std::cout << "Total Floating Point Operations: " << total_flop << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
    ip_distance_avx512(mat_a.data(), mat_b.data(), mat_b_size, mat_b_dim, engine, stream);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "IP 1 x N AVX512: " << dur << " ms" << std::endl;
    std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ip_distance_amx(mat_a.data(), mat_b.data(), mat_a_size, mat_b_size, mat_b_dim, engine, stream);
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "IP 1 x N AMX: " << dur << " ms" << std::endl;
    std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
}

void run_ip_N_x_N(dnnl::engine &engine, dnnl::stream &stream) {
    int32_t mat_a_size = CONST_A;
    int32_t mat_a_dim = CONST_A;
    int32_t mat_b_size = CONST_A;
    int32_t mat_b_dim = CONST_A;

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

    int64_t total_flop = (int64_t)mat_a_size * (int64_t)mat_b_size * (2 * (int64_t)mat_a_dim - 1);
    std::cout << "Total Floating Point Operations: " << total_flop << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < mat_a_size; i++) {
        ip_distance_avx512(mat_a.data() + i * mat_a_dim, mat_b.data(), mat_b_size, mat_b_dim, engine, stream);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "IP N x N AVX512: " << dur << " ms" << std::endl;
    std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ip_distance_amx(mat_a.data(), mat_b.data(), mat_a_size, mat_b_size, mat_b_dim, engine, stream);
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "IP N x N AMX: " << dur << " ms" << std::endl;
    std::cout << "GFLOPS: " << ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 3))) << std::endl;
}   

void run_bench() {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);
    run_ip_1_x_N(engine, stream);
    run_ip_N_x_N(engine, stream);
}

} // namespace avs

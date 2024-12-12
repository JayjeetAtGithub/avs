#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <random>
#include "dist.hpp"
#include "VariadicTable.hpp"

using pprinter = VariadicTable<std::string, std::string, double, double, double, double>;

namespace avs {

class Benchmark {
public:
    bool only_amx = false;
    dnnl::engine engine;
    dnnl::stream stream;
    pprinter *pt;
    std::vector<std::string> headers = 
        {"Mode", "N1 / N2 / M", "Data size (MiB)", "Total FLOP", "Duration (us)", "GFLOPS"};

    Benchmark(dnnl::engine engine, dnnl::stream stream) : engine(engine), stream(stream) {
        pt = new pprinter(headers);
    }

    void print_results() {
        pt->print(std::cout);
        pt = new pprinter(headers);
    }

    void run_ip(uint64_t N1, uint64_t N2, uint64_t M) {
        std::vector<float> mat_a(N1 * M);
        std::vector<float> mat_b(N2 * M);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<float> distrib;

        for (uint64_t i = 0; i < N1; i++) {
            for (uint64_t j = 0; j < M; j++) {
                mat_a[i * M + j] = distrib(rng);
            }
        }
        for (uint64_t i = 0; i < N2; i++) {
            for (uint64_t j = 0; j < M; j++) {
                mat_b[i * M + j] = distrib(rng);
            }
        }

        double data_size = ((double)(N1 * M * 4) + (double)(N2 * M * 4)) / pow(10, 6);
        uint64_t total_flop = (N1 * N2) * (2 * M - 1);
        std::string dims = std::to_string(N1) + "/" + std::to_string(N2) + "/" + std::to_string(M);

        if (!only_amx) {
            auto start = std::chrono::high_resolution_clock::now();
            for (uint64_t i = 0; i < N1; i++) {
                ip_distance_avx512(mat_a.data() + i * M, mat_b.data(), N2, M, engine, stream);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double gflops = ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 6)));
            pt->addRow("IP / AVX512", dims, data_size, total_flop, dur, gflops);
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            amx_inner_product(
                N1, N2, M, mat_a.data(), mat_b.data(), engine, stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double gflops = ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 6)));
            pt->addRow("IP / AMX", dims, data_size, total_flop, dur, gflops);
        }
    }

    void run_gemm(uint64_t N1, uint64_t N2, uint64_t M) {
        std::vector<float> mat_a(N1 * M);
        std::vector<float> mat_b(M * N2);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<float> distrib;

        for (uint64_t i = 0; i < N1; i++) {
            for (uint64_t j = 0; j < M; j++) {
                mat_a[i * M + j] = distrib(rng);
            }
        }
        for (uint64_t i = 0; i < M; i++) {
            for (uint64_t j = 0; j < N2; j++) {
                mat_b[i * N2 + j] = distrib(rng);
            }
        }

        double data_size = ((double)(N1 * M * 4) + (double)(M * N2 * 4)) / pow(10, 6);
        uint64_t total_flop = (N1 * N2) * (2 * M - 1);
        std::string dims = std::to_string(N1) + "/" + std::to_string(N2) + "/" + std::to_string(M);

        {
            auto start = std::chrono::high_resolution_clock::now();
            amx_matmul(
                N1, N2, M, mat_a.data(), mat_b.data(), engine, stream);
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            double gflops = ((double)(total_flop / pow(10, 9))) / ((double)(dur / pow(10, 6)));
            pt->addRow("GEMM / AMX", dims, data_size, total_flop, dur, gflops);
        }
    }
};

void run_bench() {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);
    
    Benchmark bench(engine, stream);

    // Just bench AMX
    bench.only_amx = true;
    std::vector<uint64_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    for (auto size : sizes) {
        bench.run_ip(size, size, size);
    }
    bench.print_results();
    for (auto size : sizes) {
        bench.run_gemm(size, size, size);
    }
    bench.print_results();
}

} // namespace avs

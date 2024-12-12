#include <iostream>
#include <vector>
#include <random>

#include "ivf.hpp"
#include "bench.hpp"
#include "CLI11.hpp"

int main(int argc, char **argv) {
    CLI::App app{"Accelerated Vector Search"};
    argv = app.ensure_utf8(argv);

    int32_t dim = 16;
    app.add_option("-d,--dim", dim, "The dimension of the vectors");

    int32_t top_k = 10;
    app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

    int32_t batch_size = 1024;
    app.add_option("-b,--batch-size", batch_size, "The batch size to use");

    int32_t num_vectors = 10'000;
    app.add_option("--nd", num_vectors, "Number of vectors in the dataset");

    int32_t num_queries = 1'000;
    app.add_option("--nq", num_queries, "Number of queries to execute");

    CLI11_PARSE(app, argc, argv);

    // Run hardware benchmarks
    avs::run_bench();

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib;


    int32_t n_list = 4 * std::sqrt(num_vectors);
    auto ivf_index = new avs::IVFFlat(n_list, 1, dim, avs::metric::IP);

    std::vector<float> data(dim * num_vectors);
    std::vector<float> queries(dim * num_queries);

    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = distrib(rng);
        }
    }

    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < dim; j++) {
            queries[i * dim + j] = distrib(rng);
        }
    }
    
    auto s = std::chrono::high_resolution_clock::now();
    ivf_index->train(data.data(), num_vectors);
    auto e = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    std::cout << "Training time (AMX): " << ms << " ms" << std::endl;

    s = std::chrono::high_resolution_clock::now();
    auto res = ivf_index->search(
        queries.data(), num_queries, data.data(), num_vectors, top_k);
    e = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    std::cout << "Search time (AMX): " << ms << " ms" << std::endl;

    s = std::chrono::high_resolution_clock::now();
    res = ivf_index->search_avx(
        queries.data(), num_queries, data.data(), num_vectors, top_k);
    e = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    std::cout << "Search time (AVX512): " << ms << " ms" << std::endl;

}

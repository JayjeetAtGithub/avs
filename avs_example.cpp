#include <iostream>
#include <vector>
#include <random>

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
}

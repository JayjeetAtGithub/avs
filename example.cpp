#include <iostream>
#include <vector>
#include <random>

#include "bf.hpp"
#include "CLI11.hpp"


int main(int argc, char **argv) {
    CLI::App app{"Accelerated Vector Search"};
    argv = app.ensure_utf8(argv);

    int32_t dim = dim;
    app.add_option("-d,--dim", dim, "The dimension of the vectors");

    int64_t top_k = 10;
    app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

    int64_t batch_size = 1024;
    app.add_option("-b,--batch-size", batch_size, "The batch size to use");

    int64_t dataset_size = 10'000;
    app.add_option("-N,--dataset-size", dataset_size, "Number of vectors in the dataset");

    CLI11_PARSE(app, argc, argv);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib;

    auto knn_index = new avs::KNNSearch(dim, batch_size);

    while (dataset_size--) {
        std::vector<float> batch;
        for (int i = 0; i < dim; i++) {
            batch.push_back(distrib(rng));
        }
        knn_index->add(batch);
    }

    auto shape = knn_index->shape();
    std::cout << "No. of vectors: " << shape.first << std::endl;
    std::cout << "Dimension: " << shape.second << std::endl;

    avs::vecf32_t query;
    for (int i = 0; i < dim; i++) query.push_back(distrib(rng));

    auto s = std::chrono::high_resolution_clock::now();
    knn_index->search(query);
    auto e = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count() << std::endl;

    auto result = knn_index->top_k(top_k);
    for (auto const &v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

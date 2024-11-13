#include <iostream>
#include <vector>
#include <random>

#include "bf.hpp"


int main(int argc, char **argv) {
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib;

    int32_t dim = 16;
    int32_t batch_size = 512;
    auto knn_index = new avs::KNNSearch(dim, batch_size);

    int32_t N = 10'000;
    
    while (N--) {
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

    auto result = knn_index->top_k(10);
    for (auto const &v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

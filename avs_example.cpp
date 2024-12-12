#include <iostream>
#include <vector>
#include <random>

#include "ivf.hpp"
#include "CLI11.hpp"


void print_matrix(std::vector<std::vector<float>> &mat) {
    size_t limit = 10;
    for (int i = 0; i < std::min(mat.size(), limit); i++) {
        for (int j = 0; j < mat[0].size(); j++) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


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
    std::cout << "Training time: " << ms << " ms" << std::endl;
    ivf_index->print_inverted_list();

    // auto knn_index = new avs::KNNSearch(dim, batch_size);

    // for (int i = 0; i < num_vectors; i++) {
    //     std::vector<float> batch;
    //     for (int j = 0; j < dim; j++) {
    //         batch.push_back(distrib(rng));
    //     }
    //     knn_index->add(batch);
    // }

    // auto shape = knn_index->shape();
    // std::cout << "No. of vectors: " << shape.first << std::endl;
    // std::cout << "Dimension of dataset vectors: " << shape.second << std::endl;

    // std::vector<std::vector<float>> queries;
    // for (int i = 0; i < num_queries; i++) {
    //     std::vector<float> query;
    //     for (int j = 0; j < dim; j++) {
    //         query.push_back(distrib(rng));
    //     }
    //     queries.push_back(query);
    // }

    // std::cout << "No. of query vectors: " << queries.size() << std::endl;
    // std::cout << "Dimension of query vectors: " << queries[0].size() << std::endl;

    // auto s = std::chrono::high_resolution_clock::now();
    // auto result = knn_index->search_l2_vanilla(queries, top_k);
    // auto e = std::chrono::high_resolution_clock::now();
    // auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count();
    // std::cout << "Duration (L2 vanilla): " << dur_ms << std::endl;
    // print_matrix(result);

    // s = std::chrono::high_resolution_clock::now();
    // result = knn_index->search_ip_amx(queries, top_k);
    // e = std::chrono::high_resolution_clock::now();
    // dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count();
    // std::cout << "Duration (IP AMX): " << dur_ms << std::endl;
    // print_matrix(result);
}

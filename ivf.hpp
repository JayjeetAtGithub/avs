#include <faiss/Clustering.h>
#include "dist.hpp"
#include <cassert>

namespace avs {
    enum metric {
        L2,
        IP
    };

    class IVFFlat {
        int32_t n_list;
        int32_t n_probe;
        float *centroids;
        metric metric_type;
        dnnl::engine _engine;
        dnnl::stream _stream;

        // Map a centroid idx with a data idx
        std::unordered_map<int32_t, std::vector<int32_t>> inverted_list;

        void _init_onednn() {
            _engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
            _stream = dnnl::stream(_engine);
            if (!is_amxbf16_supported()) {
                std::cout << "Intel AMX unavailable" << std::endl;
            }
        }

        void _create_clusters(
                const float *data,
                int32_t n_data,
                int32_t dim
            ) {
                centroids = new float[n_list * dim];
                float err = faiss::kmeans_clustering(
                    dim,
                    n_data,
                    n_list,
                    data,
                    centroids
                );

                for (int32_t i = 0; i < n_list * dim; i++) {
                    std::cout << centroids[i] << " ";
                }
                std::cout << std::endl;
            }

            void _map_to_clusters(
                const float *data,
                int32_t n_data,
                int32_t dim
            ) {
                // We do cluseter dot product data vectors
                auto res_matrix = ip_distance_amx(
                    data, centroids, n_data, n_list, dim, _engine, _stream);

                std::cout << "Res matrix: " << std::endl;
                std::cout << res_matrix.size() << " " << res_matrix[0].size() << std::endl;
                
                int32_t data_idx = 0;
                for (auto const& row : res_matrix) {
                    auto centroid_idx = std::min_element(row.begin(), row.end()) - row.begin();
                    inverted_list[centroid_idx].push_back(data_idx);
                    data_idx++;
                }
            }

        public:

            IVFFlat(int32_t n_list, int32_t n_probe, metric metric_type) 
                : n_list(n_list), n_probe(n_probe), metric_type(metric_type) {
                _init_onednn();
            }

            void train(
                const float *data,
                int32_t n_data,
                int32_t dim
            ) {
                _create_clusters(data, n_data, dim);
                _map_to_clusters(data, n_data, dim);
            }

            void print_inverted_list() {
                std::cout << inverted_list.size() << std::endl;
                for (auto const& [centroid_idx, data_idxs] : inverted_list) {
                    std::cout << "Centroid idx: " << centroid_idx << std::endl;
                    for (auto const& data_idx : data_idxs) {
                        std::cout << data_idx << " ";
                    }
                    std::cout << std::endl;
                }
            }
    };
}

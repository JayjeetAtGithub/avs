#include <faiss/Clustering.h>
#include "dist.hpp"
#include <cassert>

namespace avs {
    enum metric {
        L2,
        IP
    };

    class IVFFlat {
        int32_t _n_list;
        int32_t _n_probe;
        int32_t _dim;
        float *centroids;
        metric _metric_type;
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
                int32_t n_data
            ) {
                centroids = new float[_n_list * _dim];
                float err = faiss::kmeans_clustering(
                    _dim,
                    n_data,
                    _n_list,
                    data,
                    centroids
                );
            }

            void _map_to_clusters(
                const float *data,
                int32_t n_data
            ) {
                // We do cluseter dot product data vectors
                auto res_matrix = ip_distance_amx(
                    data, centroids, n_data, _n_list, _dim, _engine, _stream);

                std::cout << res_matrix.size() << " " << res_matrix[0].size() << std::endl;
                
                int32_t data_idx = 0;
                for (auto const& row : res_matrix) {
                    auto centroid_idx = std::min_element(row.begin(), row.end()) - row.begin();
                    inverted_list[centroid_idx].push_back(data_idx);
                    data_idx++;
                }
            }

            std::vector<std::pair<int32_t, int32_t>> _find_closest_centroids(
                const float *query,
                int32_t n_query
            ) {
                auto res_matrix = ip_distance_amx(
                    query, centroids, n_query, _n_list, _dim, _engine, _stream);

                // list of pairs of (query idx, centroid idx)
                std::vector<std::pair<int32_t, int32_t>> closest_centroids;

                for (int32_t i = 0; i < n_query; i++) {
                    auto centroid_idx = std::min_element(
                        res_matrix[i].begin(), res_matrix[i].end()) - res_matrix[i].begin();
                    closest_centroids.push_back(std::make_pair(i, centroid_idx));
                }

                return closest_centroids;
            }

        public:

            IVFFlat(int32_t n_list, int32_t n_probe, int32_t dim, metric metric_type) 
                : _n_list(n_list), _n_probe(n_probe), _dim(dim), _metric_type(metric_type) {
                _init_onednn();
            }

            void train(
                const float *data,
                int32_t n_data
            ) {
                _create_clusters(data, n_data);
                _map_to_clusters(data, n_data);
            }

            void search(const float *queries, int32_t n_query, int32_t top_k) {
                auto closest_centroids = _find_closest_centroids(queries, n_query);
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

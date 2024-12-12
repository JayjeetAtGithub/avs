#include <faiss/Clustering.h>
#include "dist.hpp"
#include <cassert>
#include <queue>

namespace avs {
    enum metric {
        L2,
        IP
    };

    class IVFFlat {
        private:
            int32_t _n_list;
            int32_t _n_probe;
            int32_t _dim;
            float *centroids;
            metric _metric_type;
            dnnl::engine _engine;
            dnnl::stream _stream;
            std::unordered_map<int32_t, std::vector<int32_t>> inverted_list;
            void _init_onednn() {
                _engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
                _stream = dnnl::stream(_engine);
                if (!is_amxbf16_supported()) {
                    std::cout << "Intel AMX unavailable" << std::endl;
                }
            }
        public:
            IVFFlat(int32_t n_list, int32_t n_probe, int32_t dim, metric metric_type) 
                : _n_list(n_list), _n_probe(n_probe), _dim(dim), _metric_type(metric_type) {
                _init_onednn();
            }

            void train(const float *data, int32_t n_data) {
                centroids = new float[_n_list * _dim];
                auto const err = faiss::kmeans_clustering(
                    _dim,
                    n_data,
                    _n_list,
                    data,
                    centroids
                );
                auto const res_matrix = ip_distance_amx(
                    data, centroids, n_data, _n_list, _dim, _engine, _stream);
                int32_t data_idx = 0;
                for (auto const& row : res_matrix) {
                    auto const cluster_idx = std::min_element(row.begin(), row.end()) - row.begin();
                    inverted_list[cluster_idx].push_back(data_idx);
                    data_idx++;
                }
            }

            std::vector<std::vector<int32_t>> search(
                const float *queries, int32_t n_query, const float *data, int32_t n_data, int32_t top_k) {
                auto const res_matrix = ip_distance_amx(
                    queries, centroids, n_query, _n_list, _dim, _engine, _stream);
                std::vector<int32_t> query_target_clusters;
                for (int32_t i = 0; i < n_query; i++) {
                    auto const cluster_idx = std::min_element(
                        res_matrix[i].begin(), res_matrix[i].end()) - res_matrix[i].begin();
                    query_target_clusters.push_back(cluster_idx);
                }
                std::vector<std::vector<int32_t>> result;
                for (int32_t i = 0; i < n_query; i++) {
                    auto const cluster_idx = query_target_clusters[i];
                    auto const data_idxs = inverted_list[cluster_idx];
                    float *data_candidates = new float[data_idxs.size() * _dim];
                    for (int32_t j = 0; j < data_idxs.size(); j++) {
                        for (int32_t k = 0; k < _dim; k++) {
                            data_candidates[j * _dim + k] = data[data_idxs[j] * _dim + k];
                        }
                    }
                    auto const res = ip_distance_amx(
                        queries + i * _dim, data_candidates, 1, data_idxs.size(), _dim, _engine, _stream)[0];
                    std::priority_queue<
                        std::pair<int32_t, int32_t>, 
                        std::vector<std::pair<int32_t, int32_t>>, 
                        std::greater<std::pair<int32_t, int32_t>>
                        > 
                        pq;
                    for (int32_t j = 0; j < res.size(); j++) {
                        pq.push(std::make_pair(res[j], j));
                    }
                    std::vector<int32_t> top_k_idxs;
                    for (int32_t j = 0; j < top_k; j++) {
                        top_k_idxs.push_back(pq.top().second);
                        pq.pop();
                    }
                    result.push_back(top_k_idxs);    
                }
                return result;
            }

            void print_inverted_list() {
                std::cout << inverted_list.size() << std::endl;
                for (auto const& [cluster_idx, data_idxs] : inverted_list) {
                    std::cout << "Cluster idx: " << cluster_idx << std::endl;
                    for (auto const& data_idx : data_idxs) {
                        std::cout << data_idx << " ";
                    }
                    std::cout << std::endl;
                }
            }
    };
}

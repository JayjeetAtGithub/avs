#include <vector>
#include <queue>

#include "types.hpp"
#include "distance.hpp"

namespace avs {

class KNNSearch {
    int32_t _dim;
    int32_t _batch_size;
    avs::matf32_t _dataset;

    dnnl::engine engine;
    dnnl::stream stream;
    std::mutex mtx;

    public:
        void init_onednn() {
            std::unique_lock<std::mutex> lock(mtx);
            engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
            stream = dnnl::stream(engine);
        }

        KNNSearch(int32_t dim, int32_t batch_size)
            : _dim(dim), _batch_size(batch_size) {
                init_onednn();
                if (!avs::is_amxbf16_supported()) {
                    std::cout << "Intel AMX unavailable" << std::endl;
                }
            }

        void add(avs::vecf32_t point) {
            _dataset.push_back(point);
        }

        std::pair<int32_t, int32_t> shape() {
            return std::make_pair(_dataset.size(), _dataset[0].size());
        }

        avs::matf32_t search_ip(matf32_t queries, int32_t top_k) {
            std::vector<std::vector<float>> results;
            std::unordered_map
                <
                    int32_t, 
                    std::priority_queue<float, std::vector<float>, std::greater<float>>
                > map;
            int32_t idx = 0;
            while (idx < _dataset.size()) {
                int32_t curr_batch_size = std::min(
                    _batch_size, (int32_t)_dataset.size() - idx);
                std::vector<std::vector<float>> curr_batch(
                    _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
                avs::matf32_t distances = avs::ip_distance(
                    queries, curr_batch, engine, stream);
                for (int32_t i = 0; i < distances.size(); i++) {
                    for (int32_t j = 0; j < distances[0].size(); j++) {
                        map[i].push(distances[i][j]);
                    }
                }
                idx += curr_batch_size;
            }

            for (int i = 0; i < queries.size(); i++) {
                std::vector<float> q_dis;
                int32_t k = top_k;
                while(k--) {
                    q_dis.push_back(map[i].top());
                    map[i].pop();
                }
                results.push_back(q_dis);
            }
            return results;
        }

        avs::matf32_t search_l2_amx(matf32_t queries, int32_t top_k) {
            std::vector<std::vector<float>> results;
            for (auto const &query : queries) {
                std::priority_queue<float, std::vector<float>, std::greater<float>> pq;
                int32_t idx = 0;
                while (idx < _dataset.size()) {
                    int32_t curr_batch_size = std::min(
                        _batch_size, (int32_t)_dataset.size() - idx);
                    std::vector<std::vector<float>> curr_batch(
                        _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
                    avs::vecf32_t distances = avs::l2_distance_amx(
                        query, curr_batch, engine, stream);
                    for (auto const &d : distances) {
                        pq.push(d);
                    }
                    idx += curr_batch_size;
                }
                std::vector<float> q_res;
                int32_t k = top_k;
                while(k--) {
                    q_res.push_back(pq.top());
                    pq.pop();
                }
                results.push_back(q_res);
            }
            return results;
        }

        avs::matf32_t search_l2_vanilla(matf32_t queries, int32_t top_k) {
            std::vector<std::vector<float>> results;
            for (auto const &query : queries) {
                std::priority_queue<float, std::vector<float>, std::greater<float>> pq;
                int32_t idx = 0;
                while (idx < _dataset.size()) {
                    int32_t curr_batch_size = std::min(
                        _batch_size, (int32_t)_dataset.size() - idx);
                    std::vector<std::vector<float>> curr_batch(
                        _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
                    avs::vecf32_t distances = avs::l2_distance_vanilla(
                        query, curr_batch, engine, stream);
                    for (auto const &d : distances) {
                        pq.push(d);
                    }
                    idx += curr_batch_size;
                }
                std::vector<float> q_res;
                int32_t k = top_k;
                while(k--) {
                    q_res.push_back(pq.top());
                    pq.pop();
                }
                results.push_back(q_res);
            }
            return results;
        }
};

} // namespace avs

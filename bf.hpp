#include <queue>
#include <vector>

#include "distance.hpp"

namespace avs {

class KNNSearch {
  int32_t _dim;
  int32_t _batch_size;
  avs::matf32_t _dataset;

  dnnl::engine engine;
  dnnl::stream stream;

public:
  void init_onednn() {
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

  void add(avs::vecf32_t point) { _dataset.push_back(point); }

  std::pair<int32_t, int32_t> shape() {
    return std::make_pair(_dataset.size(), _dataset[0].size());
  }

  // Intel AMX versions
  avs::matf32_t search_ip_amx(matf32_t queries, int32_t top_k) {
    std::vector<std::vector<float>> results(queries.size(),
                                            std::vector<float>(top_k, 0.0f));
    std::unordered_map<int32_t, std::priority_queue<float, std::vector<float>,
                                                    std::greater<float>>>
        map;
    int32_t idx = 0;
    while (idx < _dataset.size()) {
      int32_t curr_batch_size =
          std::min(_batch_size, (int32_t)_dataset.size() - idx);
      std::vector<std::vector<float>> curr_batch(
          _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
      avs::matf32_t distances =
          avs::ip_distance_amx(queries, curr_batch, engine, stream);
      for (int32_t i = 0; i < distances.size(); i++) {
        for (int32_t j = 0; j < distances[0].size(); j++) {
          map[i].push(distances[i][j]);
        }
      }
      idx += curr_batch_size;
    }

    for (int i = 0; i < queries.size(); i++) {
      int32_t k_idx = 0;
      while (k_idx < top_k) {
        results[i][k_idx++] = map[i].top();
        map[i].pop();
      }
    }
    return results;
  }

  void search_ip_amx_perf(matf32_t queries, int32_t top_k) {
    int32_t idx = 0;
    while (idx < _dataset.size()) {
      int32_t curr_batch_size =
          std::min(_batch_size, (int32_t)_dataset.size() - idx);
      std::vector<std::vector<float>> curr_batch(
          _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
      avs::matf32_t distances =
          avs::ip_distance_amx(queries, curr_batch, engine, stream);
      idx += curr_batch_size;
    }
  }

  avs::matf32_t search_l2_amx(matf32_t queries, int32_t top_k) {
    std::vector<std::vector<float>> results(queries.size(),
                                            std::vector<float>(top_k, 0.0f));
    for (int i = 0; i < queries.size(); i++) {
      std::priority_queue<float, std::vector<float>, std::greater<float>> pq;
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::l2_distance_amx(queries[i], curr_batch, engine, stream);
        for (auto const &d : distances) {
          pq.push(d);
        }
        idx += curr_batch_size;
      }
      int32_t k_idx = 0;
      while (k_idx < top_k) {
        results[i][k_idx++] = pq.top();
        pq.pop();
      }
    }
    return results;
  }

  void search_l2_amx_perf(matf32_t queries, int32_t top_k) {
    for (int i = 0; i < queries.size(); i++) {
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::l2_distance_amx(queries[i], curr_batch, engine, stream);
        idx += curr_batch_size;
      }
    }
  }

  // Vanilla versions
  avs::matf32_t search_l2_vanilla(matf32_t queries, int32_t top_k) {
    std::vector<std::vector<float>> results(queries.size(),
                                            std::vector<float>(top_k, 0.0f));
    for (int i = 0; i < queries.size(); i++) {
      std::priority_queue<float, std::vector<float>, std::greater<float>> pq;
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::l2_distance_vanilla(queries[i], curr_batch, engine, stream);
        for (auto const &d : distances) {
          pq.push(d);
        }
        idx += curr_batch_size;
      }
      int32_t k_idx = 0;
      while (k_idx < top_k) {
        results[i][k_idx++] = pq.top();
        pq.pop();
      }
    }
    return results;
  }

  void search_l2_vanilla_perf(matf32_t queries, int32_t top_k) {
    for (int i = 0; i < queries.size(); i++) {
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::l2_distance_vanilla(queries[i], curr_batch, engine, stream);
        idx += curr_batch_size;
      }
    }
  }

  void search_l2_avx512_perf(matf32_t queries, int32_t top_k) {
    for (int i = 0; i < queries.size(); i++) {
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::l2_distance_avx512(queries[i], curr_batch, engine, stream);
        idx += curr_batch_size;
      }
    }
  }

  void search_ip_vanilla_perf(matf32_t queries, int32_t top_k) {
    for (int i = 0; i < queries.size(); i++) {
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::ip_distance_vanilla(queries[i], curr_batch, engine, stream);
        idx += curr_batch_size;
      }
    }
  }

  void search_ip_avx512_perf(matf32_t queries, int32_t top_k) {
    for (int i = 0; i < queries.size(); i++) {
      int32_t idx = 0;
      while (idx < _dataset.size()) {
        int32_t curr_batch_size =
            std::min(_batch_size, (int32_t)_dataset.size() - idx);
        std::vector<std::vector<float>> curr_batch(
            _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
        avs::vecf32_t distances =
            avs::ip_distance_avx512(queries[i], curr_batch, engine, stream);
        idx += curr_batch_size;
      }
    }
  }
};

} // namespace avs

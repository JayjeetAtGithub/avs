#include <vector>

#include "types.hpp"
#include "distance.hpp"

namespace avs {

class KNNSearch {
    int32_t _dim;
    int32_t _batch_size;
    avs::matf32_t _dataset;

    public:
        KNNSearch(int32_t dim, int32_t batch_size)
            : _dim(dim), _batch_size(batch_size) {}

        void add(avs::vecf32_t point) {
            _dataset.push_back(point);
        }

        std::pair<int32_t, int32_t> shape() {
            return std::make_pair(_dataset.size(), _dataset[0].size());
        }

        void search(vecf32_t query) {
            int32_t idx = 0;
            while (idx < _dataset.size()) {
                int32_t curr_batch_size = std::min(
                    _batch_size, (int32_t)_dataset.size() - idx);
                std::cout << "searching batch of size: " << curr_batch_size << std::endl;
                std::vector<std::vector<float>> curr_batch(
                    _dataset.begin() + idx, _dataset.begin() + idx + curr_batch_size);
                auto distances = avs::l2_distance(query, curr_batch);
                idx += curr_batch_size;
            }
        }
};

} // namespace avs

#include <immintrin.h>

#include <mutex>
#include <unordered_map>

#include "utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

namespace avs {

static dnnl::engine engine;
static dnnl::stream stream;
static bool is_onednn_init = false;
static std::mutex mtx;

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static void init_onednn() {
    std::unique_lock<std::mutex> lock(mtx);
    if (is_onednn_init) {
        return;
    }
    engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream = dnnl::stream(engine);
    is_onednn_init = true;
}

inline static bool is_amxbf16_supported() {
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    return edx & (1 << 22);
}


__attribute__((constructor)) static void lib_load() {
    init_onednn();
    if (is_amxbf16_supported()) {
        std::cout << "Intel AMX BF16 Supported." << std::endl;
    } else {
        std::cout << "Intel AMX bf16 not supported. Aborting." << std::endl;
        exit(1);
    }
}

__attribute__((destructor)) static void lib_unload();

/**
 * @brief Substract two fp32 vectors using avx512 intrinsics.
 * 
 * We use AVX512 because we are using 16 * fp32 vectors
 * currently, which fit perfectly into the avx512 registers.
 */
void avx512_subtract(float *a, float* b, float *c) {
    __m512 diff, v1, v2;
    v1 = _mm512_loadu_ps(a);
    v2 = _mm512_loadu_ps(b);
    diff = _mm512_sub_ps(v1, v2);
    _mm512_store_ps(c, diff);
}

/**
 * @brief Substract a set of data vectors from a query vectors
 * using avx512. 
 * 
 * Currently, we do the subtractions sequentially, 
 * but we are looking into parallelizing it if possible.
 */
std::vector<std::vector<float>>
avx512_subtract_batch(std::vector<float> query, std::vector<std::vector<float>> data) {
    size_t N = data.size();
    size_t dim = data[0].size();
    std::vector<std::vector<float>> result(N, std::vector<float>(dim, 0.0f));
    for (int i = 0; i < N; i++) {
        float PORTABLE_ALIGN64 tmp[dim];
        avx512_subtract(query.data(), data[i].data(), tmp);
        for (int k = 0; k < dim; k++) {
            result[i][k] = tmp[k];
        }
    }
    return result;
}

std::vector<float> amx_matmul(
    const int64_t &r, const int64_t &c, std::vector<float> &m, std::vector<float> &mt) {
    std::vector<float> dst(r * r, 2.5f);

    dnnl::memory::dims a_dims = {r, c};
    dnnl::memory::dims b_dims = {c, r};
    dnnl::memory::dims c_dims = {r, r};

    // Declare fp32 input memory descriptors
    auto a_in_md = dnnl::memory::desc(a_dims, dt::f32, tag::ab);
    auto b_in_md = dnnl::memory::desc(b_dims, dt::f32, tag::ab);
    auto c_out_md = dnnl::memory::desc(c_dims, dt::f32, tag::ab);
    auto a_in_mem = dnnl::memory(a_in_md, engine);
    auto b_in_mem = dnnl::memory(b_in_md, engine);
    write_to_dnnl_memory(m.data(), a_in_mem);
    write_to_dnnl_memory(mt.data(), b_in_mem);

    // Declare bf16 compute memory descriptors
    auto a_md = dnnl::memory::desc(a_dims, dt::bf16, tag::any);
    auto b_md = dnnl::memory::desc(b_dims, dt::bf16, tag::any);

    // Declare matmul primitive
    auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_out_md);

    // Repack and convert input data
    auto a_mem = dnnl::memory(pd.src_desc(), engine);
    dnnl::reorder(a_in_mem, a_mem).execute(stream, a_in_mem, a_mem);

    auto b_mem = dnnl::memory(pd.weights_desc(), engine);
    dnnl::reorder(b_in_mem, b_mem).execute(stream, b_in_mem, b_mem);

    auto c_mem = dnnl::memory(pd.dst_desc(), engine);

    // Create the primitive
    auto prim = dnnl::matmul(pd);
    
    // Primitive arguments.
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, a_mem});
    args.insert({DNNL_ARG_WEIGHTS, b_mem});
    args.insert({DNNL_ARG_DST, c_mem});
    prim.execute(stream, args);
    stream.wait();

    read_from_dnnl_memory(dst.data(), c_mem);

    std::vector<float> res(r, 0.0f);
    for (int i = 0; i < r; i++) {
        res[i] = dst[i * r + i];
    }
    return res;
}

[[nodiscard]] static std::vector<float> l2_distance(
    avs::vecf32_t &query, avs::matf32_t &batch) {
    const int64_t batch_size = batch.size();
    const int64_t dim = batch[0].size();
    std::vector<std::vector<float>> dis_2d = avx512_subtract_batch(query, batch);
    std::vector<std::vector<float>> dis_2d_t(
        dis_2d[0].size(), std::vector<float>(dis_2d.size(), 0.0f));
    for (int i = 0; i < dis_2d_t.size(); i++) {
        for (int j = 0; j < dis_2d_t[0].size(); j++) {
            dis_2d_t[i][j] = dis_2d[j][i];
        }
    }
    std::vector<float> dis_1d(batch_size * dim);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            dis_1d[i * dim + j] = dis_2d[i][j];
        }
    }
    std::vector<float> dis_1d_t(batch_size * dim);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < batch_size; j++) {
            dis_1d_t[i * batch_size + j] = dis_2d_t[i][j];
        }
    }
    return amx_matmul(batch_size, dim, dis_1d, dis_1d_t);
}

// [[nodiscard]] static std::vector<float> l2_distance(
//     avs::vecf32_t &query, avs::matf32_t &batch) {
//     const int64_t batch_size = batch.size();
//     const int64_t dim = batch[0].size();
//     std::vector<std::vector<float>> dis_2d = avx512_subtract_batch(query, batch);
    
   

// }

} // namespace avs

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

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

inline static bool is_amxbf16_supported() {
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    return edx & (1 << 22);
}

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
    int32_t N = data.size();
    int32_t dim = data[0].size();
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
    const int64_t &r, const int64_t &c,
    std::vector<float> &m, std::vector<float> &mt,
    dnnl::engine &engine, dnnl::stream &stream) {
    std::vector<float> dst(r * r, 0.0f);

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


/**
 * @brief Calculate the inner product between
 * two matrixes batch by batch. 
 */
std::vector<std::vector<float>> amx_inner_product(
    const int32_t &n, const int32_t &oc, const int32_t &ic,
    std::vector<float> &s, std::vector<float> &w,
    dnnl::engine &engine, dnnl::stream &stream) {

    dnnl::memory::dims s_dims = {n, ic};
    dnnl::memory::dims w_dims = {oc, ic};
    dnnl::memory::dims dst_dims = {n, oc};

    // Declare fp32 input memory descriptors
    auto s_in_md = dnnl::memory::desc(s_dims, dt::f32, tag::ab);
    auto w_in_md = dnnl::memory::desc(w_dims, dt::f32, tag::ab);
    auto dst_out_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);
    auto s_in_mem = dnnl::memory(s_in_md, engine);
    auto w_in_mem = dnnl::memory(w_in_md, engine);
    write_to_dnnl_memory(s.data(), s_in_mem);
    write_to_dnnl_memory(w.data(), w_in_mem);

    // Declare bf16 compute memory descriptors
    auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::any);
    auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::any);

    // Declare inner product primitive
    auto pd = dnnl::inner_product_forward::primitive_desc(
        engine,
        dnnl::prop_kind::forward_training, 
        s_md, 
        w_md,
        dst_out_md);

    // Repack and convert input data
    auto s_mem = dnnl::memory(pd.src_desc(), engine);
    dnnl::reorder(s_in_mem, s_mem).execute(stream, s_in_mem, s_mem);
    auto w_mem = dnnl::memory(pd.weights_desc(), engine);
    dnnl::reorder(w_in_mem, w_mem).execute(stream, w_in_mem, w_mem);
    auto dst_mem = dnnl::memory(pd.dst_desc(), engine);

    // Create the primitive
    auto prim = dnnl::inner_product_forward(pd);
    
    // Primitive arguments
    std::unordered_map<int, dnnl::memory> args;
    args.insert({DNNL_ARG_SRC, s_mem});
    args.insert({DNNL_ARG_WEIGHTS, w_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    prim.execute(stream, args);
    stream.wait();

    std::vector<float> dst(n * oc, 0.0f);
    read_from_dnnl_memory(dst.data(), dst_mem);
    std::vector<std::vector<float>> res(n, std::vector<float>(oc, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < oc; j++) {
            res[i][j] = dst[i * oc + j];
        }
    }
    return res;
}

[[nodiscard]] static std::vector<std::vector<float>> ip_distance(
    avs::matf32_t &queries, avs::matf32_t &batch, dnnl::engine &engine, dnnl::stream &stream) {

    const int32_t n = queries.size();
    const int32_t oc = batch.size();
    const int32_t ic = queries[0].size();

    std::vector<float> queries_1d(n * ic);
    std::vector<float> batch_1d(oc * ic);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < ic; j++) {
            queries_1d[i * ic + j] = queries[i][j];
        }
    }
    for (int i = 0; i < oc; i++) {
        for (int j = 0; j < ic; j++) {
            batch_1d[i * ic + j] = batch[i][j];
        }
    }

    return amx_inner_product(
        n, oc, ic,
        queries_1d, batch_1d, 
        engine, stream
    );
}


[[nodiscard]] static std::vector<float> l2_distance_amx(
    const avs::vecf32_t &query, avs::matf32_t &batch, dnnl::engine &engine, dnnl::stream &stream) {
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
    return amx_matmul(batch_size, dim, dis_1d, dis_1d_t, engine, stream);
}

static float L2Sqr(const void *vec1, const void *vec2, const int32_t dim) {
    float *v1 = (float *) vec1;
    float *v2 = (float *) vec2;

    float res = 0;
    for (size_t i = 0; i < dim; i++) {
        float t = *v1 - *v2;
        v1++;
        v2++;
        res += t * t;
    }
    return (res);
}

[[nodiscard]] static std::vector<float> l2_distance_vanilla(
    const avs::vecf32_t &query, avs::matf32_t &batch, dnnl::engine &engine, dnnl::stream &stream) {
    const int64_t dim = batch[0].size();

    std::vector<float> res;
    for (int i = 0; i < batch.size(); i++) {
        auto d = L2Sqr(query.data(), batch[i].data(), dim);
        res.push_back(d);
    }
    return res;
}

} // namespace avs

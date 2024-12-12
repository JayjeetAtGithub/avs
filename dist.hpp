#pragma once

#include <immintrin.h>
#include <chrono>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

namespace avs {

using vecf32_t = std::vector<float>;
using matf32_t = std::vector<std::vector<float>>;
using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static bool is_amxbf16_supported() {
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "a"(7), "c"(0));
  return edx & (1 << 22);
}

static void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
  if (!src)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    ((uint8_t *)handle)[i] = src[i];
  }
}

static void write_to_dnnl_memory(void const *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  int32_t size = mem.get_desc().get_size();
  if (!handle)
    throw std::runtime_error("handle is nullptr.");
  uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
  if (!dst)
    throw std::runtime_error("get_data_handle returned nullptr.");
  for (int32_t i = 0; i < size; ++i) {
    dst[i] = ((uint8_t *)handle)[i];
  }
}

static void amx_matmul(int32_t const &r1, int32_t const &r2, const int32_t &c,
                      const float *a, const float *b,
                      dnnl::engine &engine, dnnl::stream &stream) {
  avs::vecf32_t dst(r1 * r2, 0.0f);
  dnnl::memory::dims a_dims = {r1, c};
  dnnl::memory::dims b_dims = {c, r2};
  dnnl::memory::dims c_dims = {r1, r2};

  auto a_in_md = dnnl::memory::desc(a_dims, dt::f32, tag::ab);
  auto b_in_md = dnnl::memory::desc(b_dims, dt::f32, tag::ab);
  auto c_out_md = dnnl::memory::desc(c_dims, dt::f32, tag::ab);
  auto a_in_mem = dnnl::memory(a_in_md, engine);
  auto b_in_mem = dnnl::memory(b_in_md, engine);
  write_to_dnnl_memory(a, a_in_mem);
  write_to_dnnl_memory(b, b_in_mem);

  auto a_md = dnnl::memory::desc(a_dims, dt::bf16, tag::any);
  auto b_md = dnnl::memory::desc(b_dims, dt::bf16, tag::any);
  auto c_md = dnnl::memory::desc(c_dims, dt::bf16, tag::any);
  auto pd = dnnl::matmul::primitive_desc(engine, a_md, b_md, c_md);

  auto a_mem = dnnl::memory(pd.src_desc(), engine);
  dnnl::reorder(a_in_mem, a_mem).execute(stream, a_in_mem, a_mem);
  auto b_mem = dnnl::memory(pd.weights_desc(), engine);
  dnnl::reorder(b_in_mem, b_mem).execute(stream, b_in_mem, b_mem);
  auto c_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::matmul(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, a_mem});
  args.insert({DNNL_ARG_WEIGHTS, b_mem});
  args.insert({DNNL_ARG_DST, c_mem});
  prim.execute(stream, args);
  stream.wait();
}

static void amx_inner_product(int32_t const &n, int32_t const &oc,
                              int32_t const &ic, const float *s,
                              const float* w, dnnl::engine &engine,
                              dnnl::stream &stream) {

  dnnl::memory::dims s_dims = {n, ic};
  dnnl::memory::dims w_dims = {oc, ic};
  dnnl::memory::dims dst_dims = {n, oc};

  auto s_in_md = dnnl::memory::desc(s_dims, dt::f32, tag::ab);
  auto w_in_md = dnnl::memory::desc(w_dims, dt::f32, tag::ab);
  auto dst_out_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);
  auto s_in_mem = dnnl::memory(s_in_md, engine);
  auto w_in_mem = dnnl::memory(w_in_md, engine);
  write_to_dnnl_memory(s, s_in_mem);
  write_to_dnnl_memory(w, w_in_mem);

  auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::any);
  auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::any);
  auto dst_md = dnnl::memory::desc(dst_dims, dt::bf16, tag::any);

  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_training, s_md, w_md, dst_md);

  auto s_mem = dnnl::memory(pd.src_desc(), engine);
  dnnl::reorder(s_in_mem, s_mem).execute(stream, s_in_mem, s_mem);
  auto w_mem = dnnl::memory(pd.weights_desc(), engine);
  dnnl::reorder(w_in_mem, w_mem).execute(stream, w_in_mem, w_mem);
  auto dst_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::inner_product_forward(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, s_mem});
  args.insert({DNNL_ARG_WEIGHTS, w_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  prim.execute(stream, args);
  stream.wait();
}

static float inner_product(void const *vec1, void const *vec2,
                          int32_t const &dim) {
  float *v1 = (float *)vec1;
  float *v2 = (float *)vec2;
  float result = 0;
  for (int32_t i = 0; i < dim; i++) {
    result += ((float *)v1)[i] * ((float *)v2)[i];
  }
  return result;
}

static float
inner_product_avx512(const void *vec1, const void *vec2, int32_t const &dim) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) vec1;
    float *pVect2 = (float *) vec2;

    size_t qty16 = dim / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static avs::vecf32_t ip_distance_avx512(const float* query,
                                        const float* data,
                                        int32_t data_size,
                                        int32_t dim,
                                        dnnl::engine &engine,
                                        dnnl::stream &stream) {
  avs::vecf32_t result(data_size);
  for (int32_t i = 0; i < data_size; i++) {
    auto d = inner_product_avx512(query, data + i * dim, dim);
    result[i] = d;
  }
  return result;
}

} // namespace avs

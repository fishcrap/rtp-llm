#pragma once
#include "../type_bf16/hie_bfloat16.hpp"

namespace fastertransformer {

template<typename Ta, typename Tb, typename Tp>
class GemmPartParam {
public:
    GemmPartParam(int    M,
                  int    N,
                  int    K_pack,
                  Ta*    a_ptr,
                  Tb*    b_ptr,
                  float* c_ptr,
                  float* bias_ptr,
                  Tp*    wei_scale,
                  Tp*    wei_scaleXzp,
                  int    GroupSize,
                  int    with_bias,
                  int    actType):
        M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(wei_scale),
        wei_scaleXzp(wei_scaleXzp),
        GroupSize(GroupSize),
        with_bias(with_bias),
        actType(actType) {}

    GemmPartParam(
        int M, int N, int K_pack, Ta* a_ptr, Tb* b_ptr, float* c_ptr, float* bias_ptr, int with_bias, int actType):
        M(M),
        N(N),
        K_pack(K_pack),
        a_ptr(a_ptr),
        b_ptr(b_ptr),
        c_ptr(c_ptr),
        bias_ptr(bias_ptr),
        wei_scale(nullptr),
        wei_scaleXzp(nullptr),
        GroupSize(K_pack),
        with_bias(with_bias),
        actType(actType) {}

    GemmPartParam() {}
    ~GemmPartParam() {}

    int    M, N, K_pack;
    Ta*    a_ptr;
    Tb*    b_ptr;
    float* c_ptr;
    float* bias_ptr;
    Tp *   wei_scale, *wei_scaleXzp;
    int    GroupSize;
    int    with_bias;
    int    actType = 0;
    int    do_act  = 1;
};

enum UnaryType : int {
    UNARYTYPE_UNDEFINED = 0,
    TANH = 1,
    GELU_ERF = 2,
    GELU_TANH = 3,
    RELU = 4,
    SILU = 5,
    UnaryType_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::min(),
    UnaryType_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::max()
};

class GemmKernel {
private:
    void pack_input_impl_parallel_simd(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16);

    void thread_block_bf16_m8(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_mres(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_nres(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile);
    void
    thread_block_bf16_m8_res(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile);

    void pack_input_arm(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16);


    void gemm_thread_block_bf16(
        GemmPartParam<hie::bfloat16, hie::bfloat16, float> p, int m, int n, int m_tile, int n_tile, int k_tile);

    void gemm_thread_strategy(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p);

public:
    void gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack, const float* b_fp32, hie::bfloat16* b_bf16);

    void gemm_kernel_arm(int            M,
                         int            N,
                         int            K,
                         int            lda,
                         float*         a_fp32,
                         hie::bfloat16* b_bf16,
                         float*         c_fp32,
                         float*         bias_fp32,
                         int            actType,
                         void*          workspace);
};

}  // namespace fastertransformer
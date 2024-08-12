#include <arm_sve.h>
#include <cstring>
#include <type_traits>

#include "ArmGemmKernel.h"
#include "gemm_microkernel_macro_m8_bf16.h"
#include "activation_const.hpp"
#include "arm_common.h"
namespace fastertransformer {

void GemmKernel::gemm_thread_block_bf16(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float> p, int m, int n, int m_tile, int n_tile, int k_tile) {
    int nn_max     = std::min(p.N, (n + 1) * n_tile);
    int mm_max     = std::min(p.M, (m + 1) * m_tile);
    int last_block = 0;
    if (mm_max == p.M && p.M % 8 != 0)
        last_block |= 0x1;

    if (nn_max == p.N && p.N % 8 != 0)
        last_block |= 0x2;

    for (int k = 0; k < p.K_pack; k += k_tile) {
        p.do_act = 0;
        if ((k + k_tile) >= p.K_pack)
            p.do_act = 0;  // TODO: change back to 1
        for (int nn = n * n_tile; nn < nn_max; nn += 8) {
            for (int mm = m * m_tile; mm < mm_max; mm += 8) {
                if (LIKELY(last_block == 0x0)) {
                    thread_block_bf16_m8(p, mm, nn, k, k_tile);
                } else if (last_block == 0x1) {
                    thread_block_bf16_m8_mres(p, mm, nn, k, k_tile);
                } else if (last_block == 0x2) {
                    thread_block_bf16_m8_nres(p, mm, nn, k, k_tile);
                } else {
                    thread_block_bf16_m8_res(p, mm, nn, k, k_tile);
                }
            }
        }
    }
}

void GemmKernel::gemm_thread_strategy(GemmPartParam<hie::bfloat16, hie::bfloat16, float, float>& p) {
    int m_tile = 32;
    int n_tile = 64;
    int k_tile = 2560;
    if (p.K_pack == 5120)
        k_tile = 5120;

    int m_max = (p.M + m_tile - 1) / m_tile;
    int n_max = (p.N + n_tile - 1) / n_tile;
    parallel_for(m_max, n_max, [&](int m, int n) {
        gemm_thread_block_bf16(p, m, n, m_tile, n_tile, k_tile);
    });
    return;
}

void GemmKernel::gemm_kernel_arm(int            M,
                                 int            N,
                                 int            K,
                                 int            k_pack,
                                 int            lda,
                                 float*         a_fp32,
                                 hie::bfloat16* b_bf16,
                                 float*         c_fp32,
                                 float*         bias_fp32,
                                 int            actType,
                                 void*          workspace) {

// #ifdef GEMM_DEBUG
//     std::cout << "gemm_thread_strategy: M=" << M << ", N=" << N << ", K=" << K << ", lda=" << lda
//               << ", actType=" << actType << "\n";
// #endif
    int K_pack    = k_pack;
    int with_bias = bias_fp32 == nullptr ? 0 : 1;

    hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
    // int            a_bf16_size = (M * K_pack + M % 2 * K_pack) * 2;  // 括号内确保对齐需要额外增加的存储空间，M是奇数的时候多加一行K_pack, * 2是因为sizeof(bf16) = 2
    // memset(a_bf16, 0, a_bf16_size);

    pack_input_arm(M, N, K, lda, K_pack, a_fp32, a_bf16);

    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float> p(
        M, N, k_pack, a_bf16, b_bf16, c_fp32, bias_fp32, with_bias, actType);

    gemm_thread_strategy(p);
    return;
}

void GemmKernel::gemm_kernel_arm(int            M,
                                 int            N,
                                 int            K,
                                 int            k_pack,
                                 int            lda,
                                 float16_t*     a_fp16,
                                 hie::bfloat16* b_bf16,
                                 float*         c_fp32,
                                 float*         bias_fp32,
                                 int            actType,
                                 void*          workspace) {

// #ifdef GEMM_DEBUG
//     std::cout << "gemm_thread_strategy: M=" << M << ", N=" << N << ", K=" << K << ", lda=" << lda
//               << ", actType=" << actType << "\n";
// #endif
    int K_pack    = k_pack;
    int with_bias = bias_fp32 == nullptr ? 0 : 1;

    hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
    // int            a_bf16_size = (M * k_pack_mem + M % 2 * k_pack_mem) * 2;  // 括号内确保对齐需要额外增加的存储空间，M是奇数的时候多加一行K_pack, * 2是因为sizeof(bf16) = 2
    // memset(a_bf16, 0, a_bf16_size);

    pack_input_fp16tobf16_impl_parallel_simd(M, N, K, lda, K_pack, a_fp16, a_bf16);

    GemmPartParam<hie::bfloat16, hie::bfloat16, float, float> p(
        M, N, K_pack, a_bf16, b_bf16, c_fp32, bias_fp32, with_bias, actType);

    gemm_thread_strategy(p);
    return;
}

}  // namespace fastertransformer
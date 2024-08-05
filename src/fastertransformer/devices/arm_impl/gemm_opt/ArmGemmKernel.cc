#include <arm_sve.h>
#include <cstring>

#include "ArmGemmKernel.h"
#include "gemm_microkernel_macro_m8_bf16.h"
#include "activation_const.hpp"
#include "arm_common.h"

namespace fastertransformer {

void GemmKernel::gemm_thread_block_bf16(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float> p, int m, int n, int m_tile, int n_tile, int k_tile) {
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

void GemmKernel::gemm_thread_strategy(GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p) {
    int m_tile = 32;
    int n_tile = 64;
    int k_tile = 2560;
    if (p.K_pack == 5120)
        k_tile = 5120;

    int m_max = (p.M + m_tile - 1) / m_tile;
    int n_max = (p.N + n_tile - 1) / n_tile;
    parallel_for(m_max, n_max, [&](int m, int n) { gemm_thread_block_bf16(p, m, n, m_tile, n_tile, k_tile); });
    return;
}

void GemmKernel::gemm_kernel_arm(int            M,
                                 int            N,
                                 int            K,
                                 int            lda,
                                 float*         a_fp32,
                                 hie::bfloat16* b_bf16,
                                 float*         c_fp32,
                                 float*         bias_fp32,
                                 int            actType,
                                 void*          workspace) {

    std::cout << "gemm_thread_strategy: M=" << M << ", N=" << N << ", K=" << K << ", lda=" << lda
              << ", actType=" << actType << "\n";
    int K_pack    = std::ceil(K / 8.0) * 8;  // k 向上取整到8的倍数
    int with_bias = bias_fp32 == nullptr ? 0 : 1;

    hie::bfloat16* a_bf16 = reinterpret_cast<hie::bfloat16*>(workspace);
    int            a_bf16_size =
        (M * K_pack + M % 2 * K_pack)
        * 2;  // 括号内确保对齐需要额外增加的存储空间，M是奇数的时候多加一行K_pack, * 2是因为sizeof(bf16) = 2
    memset(a_bf16, 0, a_bf16_size);

    pack_input_arm(M, N, K, lda, K_pack, a_fp32, a_bf16);

    GemmPartParam<hie::bfloat16, hie::bfloat16, float> p(
        M, N, K_pack, a_bf16, b_bf16, c_fp32, bias_fp32, with_bias, actType);

    gemm_thread_strategy(p);
    return;
}

void GemmKernel::pack_input_arm(int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16) {
    pack_input_impl_parallel_simd(M, N, K, lda, K_pack, a_fp32, a_bf16);
    return;
}

void GemmKernel::gemm_pack_weight_FP32toBF16_arm(int N, int K, int K_pack, const float* b_fp32, hie::bfloat16* b_bf16) {
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K_pack * 1.0 / k_tile);

    parallel_for(k_thread, [&](int k) {
        for (int n = 0; n < N; n += 2) {
            float*         b_fp32_ptr1 = (float*)b_fp32 + k * k_tile * N + n + 0;
            float*         b_fp32_ptr2 = (float*)b_fp32 + k * k_tile * N + n + 1;
            hie::bfloat16* b_bf16_ptr  = b_bf16 + n * K_pack + k * k_tile * 2;
            int            kk_max      = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;
            for (int kk = k * k_tile; kk < kk_max; kk += 4) {
                for (int i = 0; i < 4 && (kk + i < kk_max); i++) {
                    b_bf16_ptr[i] = b_fp32_ptr1[i * N];
                    if (n != (N - 1)) {
                        b_bf16_ptr[i + 4] = b_fp32_ptr2[i * N];
                    }
                }
                b_bf16_ptr += 8;
                b_fp32_ptr1 += 4 * N;
                b_fp32_ptr2 += 4 * N;
            }
        }
    });

    return;
}

void GemmKernel::pack_input_impl_parallel_simd(
    int M, int N, int K, int lda, int K_pack, float* a_fp32, hie::bfloat16* a_bf16) {
#define LABEL_FOR_LOOP_M "0"
#define LABEL_FOR_LOOP_K "1"
#define LABEL_m_EQ_M_1 "2"
    int k_tile   = 1024;  // empirical var: 1024, 5120
    int k_thread = std::ceil(K * 1.0 / k_tile);

    // printf("k_tile: %d, k_thread: %d\n", k_tile, k_thread);

    // fp32 [ a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3] ]
    // fp32 [ a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3] ]
    // bf16 [ a[i+1,j+0], a[i+1,j+1], a[i+1,j+2], a[i+1,j+3],
    //        a[i,  j+0], a[i,  j+1], a[i,  j+2], a[i,  j+3]] ???

    parallel_for(k_thread, [&](int k) {
        float*         a_fp32_ptr1   = a_fp32 + 0 * lda + k * k_tile;
        float*         a_fp32_ptr2   = a_fp32 + 1 * lda + k * k_tile;
        hie::bfloat16* a_bf16_ptr    = a_bf16 + k * k_tile * 2;
        int            a_fp32_offset = 2 * lda * sizeof(float);
        int            a_bf16_offset = 2 * K_pack * sizeof(hie::bfloat16);
        int            kk            = k * k_tile;
        int            kk_max        = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;

        // clang-format off
        asm volatile(
            "ptrue   p0.b                                    \n"
            "sub     x1,    %[M], #1                         \n"  // M - 1
            "mov     x2,    #0                               \n"  // m

            "" LABEL_FOR_LOOP_M
            ":\n"
            "mov     x3,    %[a_fp32_ptr1]                   \n"
            "mov     x4,    %[a_fp32_ptr2]                   \n"
            "mov     x5,    %[a_bf16_ptr]                    \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "mov     x0,    %[kk]                            \n"
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max

            "" LABEL_FOR_LOOP_K
            ":\n"
            "ld1w   z0.s, p1/z, [x3,    #0, MUL VL]          \n"
            "dup    z1.h, #0                                 \n"
            "cmp    x2, x1                                   \n"  // compare m,
                                                                  // M - 1
            "b.none  " LABEL_m_EQ_M_1
            "f                     \n"
            "ld1w   z1.s, p1/z, [x4,    #0, MUL VL]          \n"  // load, when
                                                                  // m != M - 1

            "" LABEL_m_EQ_M_1
            ":\n"
            "add     x3, x3, #16                             \n"
            "add     x4, x4, #16                             \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "bfcvt   z0.h, p0/m, z0.s                        \n"  // fp32 ->
                                                                  // bf16
            "bfcvt   z1.h, p0/m, z1.s                        \n"
            "uzp1    z2.h, z0.h, z1.h                        \n"  // combine
                                                                  // bf16

            "uzp1    p3.h, p1.h, p1.h                        \n"
            "st1h    z2.h, p3,   [x5, #0, MUL VL]            \n"  // store bf16
                                                                  // data
            "add     x5, x5, #16                             \n"

            //   "prfw    pstl1keep, p0, [x5,    #0, MUL VL]      \n"

            "add     x0,    x0,   #4                         \n"  // kk += 4
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max
            "b.tstop " LABEL_FOR_LOOP_K
            "b                   \n"  // if k < K_MAX, go to label

            "add     %[a_fp32_ptr1], %[a_fp32_ptr1], %[a_fp32_offset] \n"
            "add     %[a_fp32_ptr2], %[a_fp32_ptr2], %[a_fp32_offset] \n"
            "add     %[a_bf16_ptr],  %[a_bf16_ptr],  %[a_bf16_offset] \n"
            "add     x2,    x2,   #2                         \n"  // m += 2
            "cmp     x2, %[M]                                \n"  // compare m,
                                                                  // M
            "b.tstop " LABEL_FOR_LOOP_M
            "b                   \n"  // if m < M, go to label

            : /* empty OutputOperands */
            : [a_fp32_ptr1] "r"(a_fp32_ptr1), [a_fp32_ptr2] "r"(a_fp32_ptr2),
              [a_bf16_ptr] "r"(a_bf16_ptr), [kk] "r"(kk), [kk_max] "r"(kk_max),
              [M] "r"(M), [a_fp32_offset] "r"(a_fp32_offset),
              [a_bf16_offset] "r"(a_bf16_offset)
            : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "p1", "p2", "p3", "z0",
              "z1", "z2", "cc", "memory");
        // clang-format on
    });

    return;
}

void GemmKernel::thread_block_bf16_m8(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

  asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0
        
        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                       \n"
        "whilelt p5.h,  x4,   %[K_MAX]           \n" // compare k and K_MAX
        "add     x4,    x4,   #8                 \n"
        "whilelt p4.h,  x4,   %[K_MAX]           \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", 
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
    return;
}

/*********************************************************/

void GemmKernel::thread_block_bf16_m8_mres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

  asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"

        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_RES_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_RES_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

/*********************************************************/

void GemmKernel::thread_block_bf16_m8_nres(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;  // 2 --> sizeof(bfloat16)
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

  asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        // "mov     x7, #0                             \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x4,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N
        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
        ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
        ASM_BLOCK_C_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

    ASM_BLOCK_C_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

/*********************************************************/

void GemmKernel::thread_block_bf16_m8_res(
    GemmPartParam<hie::bfloat16, hie::bfloat16, float>& p, int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_SKIP_PRF "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_W1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

    int M = p.M;
    int N = p.N;

    hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr3 = p.a_ptr + (m + 4) * p.K_pack + k * 2;
    hie::bfloat16* a_bf16_ptr4 = p.a_ptr + (m + 6) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr3 = p.b_ptr + (n + 4) * p.K_pack + k * 2;
    hie::bfloat16* b_bf16_ptr4 = p.b_ptr + (n + 6) * p.K_pack + k * 2;

    uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

    int next_line_offset = N * sizeof(float);

    float* bias_ptr = p.bias_ptr + n;

    int k_init = k * 2;
    int K_MAX  = (k + k_tile) * 2;
    K_MAX      = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;
    int K_MAIN = K_MAX / 16 * 16;

    activation_const_t constant;

    // clang-format off

  asm volatile(
        "ptrue   p0.b                               \n"
        "ptrue   p4.b                               \n"
        "ptrue   p5.b                               \n"

        // ASM_BLOCK_PREFETCH_PART_0

        "mov     x0, %[k_init]                      \n" // k
        "mov     x2, %[m]                           \n"
        "mov     x3, %[n]                           \n"

        /* clear bfmmla result regs */
        ASM_BLOCK_CLEAR_BFMMLA_REG

        " " LABEL_FOR_LOOP_K ":\n"
        /* load bf16 input & weight */
        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        "add     x0,    x0,   #16                \n" // k += 16
        "cmp     x0,    %[K_MAIN]                \n" // compare k and K_MAIN
        "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAIN, go to label

        /* load bf16 input & weight */
        "mov     x4,    x0                           \n"
        "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
        "add     x4,    x0,   #8                     \n"
        "whilelt p4.h,  x4,   %[K_MAX]               \n"

        ASM_BLOCK_LOAD_A_RES
        ASM_BLOCK_LOAD_B_RES

        // ASM_BLOCK_PREFETCH_PART_1

        /* matmul */
        ASM_BLOCK_BFMMLA

        /* reorder mmla output */
        ASM_BLOCK_REORDER_BFMMLA_OUTPUT

        "whilelt p1.s, x3, %[N]                  \n" // compare n, N
        "add     x6,   x3, #4                    \n" // n + 2
        "whilelt p2.s, x6, %[N]                  \n" // compare n, N

        : /* empty OutputOperands */
        : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [a_bf16_ptr3] "r"(a_bf16_ptr3), [a_bf16_ptr4] "r"(a_bf16_ptr4),
        [b_bf16_ptr1] "r"(b_bf16_ptr1), [b_bf16_ptr2] "r"(b_bf16_ptr2),
        [b_bf16_ptr3] "r"(b_bf16_ptr3), [b_bf16_ptr4] "r"(b_bf16_ptr4),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAIN] "r"(K_MAIN), [K_MAX] "r"(K_MAX)
        : "p0", "p1", "p2", "p4", "p5", 
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "z30", "z31", 
        "cc", "memory");

    if (p.with_bias && k == 0) {
    ASM_BLOCK_ADD_BIAS
    }

    if (LIKELY(k != 0)) {
    ASM_BLOCK_C_RES_ACCUMULATE
    }

    if (p.do_act == 1) {
        switch (p.actType) {
            case UnaryType::UNARYTYPE_UNDEFINED: {
                break;
            }
            case UnaryType::RELU: {
                ASM_BLOCK_ACTIVE_RELU
                break;
            }
            case UnaryType::SILU: {
                ASM_BLOCK_ACTIVE_SILU
                break;
            }
            case UnaryType::TANH: {
                ASM_BLOCK_ACTIVE_TANH
                break;
            }
            case UnaryType::GELU_ERF: {
                ASM_BLOCK_ACTIVE_GELU_ERF
                break;
            }
            case UnaryType::GELU_TANH: {
                ASM_BLOCK_ACTIVE_GELU_TANH
                break;
            }
            default:
                break;
        }
    }

  ASM_BLOCK_C_RES_STORE

    // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_SKIP_PRF
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_W1
#undef LABEL_SKIP_ACCUMULATE
    return;
}

}  // namespace fastertransformer
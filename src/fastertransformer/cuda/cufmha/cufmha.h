#pragma once

#include "3rdparty/flash_attention/flash_api.h"
#include "3rdparty/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "3rdparty/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "3rdparty/trt_fused_multihead_attention/qkvToContext.h"
#include "src/fastertransformer/core/Types.h"

namespace fastertransformer{

class cufmha {

public:
    cufmha() = default;
    ~cufmha() = default;

    void init(cudaStream_t stream) {
        stream_ = stream;
    }

    void setup(DataType dtype,
               AttentionMaskType mtype,
               size_t head_num,
               size_t kv_head_num,
               size_t size_per_head,
               float q_scaling,
               bool  use_linear_bias_slopes)
    {
        dtype_ = dtype;
        mtype_ = mtype;
        head_num_ = head_num;
        kv_head_num_ = kv_head_num;
        size_per_head_ = size_per_head;
        q_scaling_ = q_scaling;
        use_linear_bias_slopes_ = use_linear_bias_slopes;
    }


    bool trtV1FmhaSupport();

    bool trtV2FmhaSupport();

    bool openSourceFmhaSupport();

    void runTrtV1Fmha(void* input,
                      void* cu_seqlens,
                      void* output,
                      void* qkv_buf_temp,
                      size_t batch_size,
                      size_t seq_len,
                      size_t token_num);

    void runTrtV2Fmha(void* input,
                      void* cu_seqlens,
                      void* output,
                      size_t batch_size,
                      size_t seq_len,
                      size_t token_num,
                      bool mFMHAForceFP32Acc    = false,
                      bool mRemovePadding       = false,
                      bool is_alibi             = false,
                      bool is_alibi_with_sacle  = false);

    void runTrtV2FmhaPaged(void*  input,
                           void*  cu_q_seqlens,
                           void*  cu_kv_seqlens,
                           void*  output,
                           size_t batch_size,
                           size_t input_seq_len,
                           size_t max_past_kv_len,
                           size_t token_num,
                           KVBlockArray kv_block_array,
                           bool mFMHAForceFP32Acc    = false,
                           bool mRemovePadding       = false,
                           bool is_alibi             = false,
                           bool is_alibi_with_sacle  = false);

    void runOpenSourceFmha(void*  q,
                           void*  k,
                           void*  v,
                           void*  output,
                           int*   cu_seqlens,
                           size_t batch_size,
                           size_t seq_len,
                           void   *workspace,
                           float* linear_bias_slopes = nullptr);

    void runOpenSourceFmhaPaged(void*  q,
                                void*  k,
                                void*  v,
                                void*  output,
                                int*   cu_seqlens,
                                int*   cu_kv_seqlens,
                                int*   block_table,
                                size_t batch_size,
                                size_t block_table_batch_stride,
                                size_t seq_size_per_block,
                                size_t seq_len,
                                void   *workspace,
                                float* linear_bias_slopes = nullptr);

    size_t getOpenSourceWorkSpaceSize(size_t batch_size,
                                      size_t seq_len_q,
                                      size_t max_seq_len_kv = 0,
                                      bool   paged = false);
private:
    static int roundMultiple(int x, int m) {
        return (x + m - 1) / m * m;
    }

    int getNumSplits(size_t batch_size,
                     size_t seqlen_q,
                     size_t seqlen_k) const;

    Flash_fwd_params genFlashFwdParams(void* q,
                                       void* k,
                                       void* v,
                                       void* output,
                                       int* cu_seqlens,
                                       int* cu_kv_seqlens,
                                       void* softmax_lse,
                                       size_t batch_size,
                                       size_t seq_len_q,
                                       size_t seq_len_kv,
                                       float* linear_bias_slopes = nullptr) const;
private:

    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> trtv2_fmha_runner_;
#ifdef USE_OLD_TRT_FMHA
    std::unique_ptr<FusedMHARunnerFP16v2> trtv1_fmha_runner_;
#endif
    DataType dtype_;
    AttentionMaskType mtype_;

    size_t head_num_;
    size_t kv_head_num_;
    size_t size_per_head_;
    float q_scaling_;
    bool use_linear_bias_slopes_;

    cudaStream_t stream_;
};

} // namespace fastertransformer

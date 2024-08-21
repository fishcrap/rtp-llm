#include "cufmha.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/utils/compiler_config.h"

namespace fastertransformer{

tensorrt_llm::kernels::Data_type trtDtypeConvert(DataType dtype)
{
    switch (dtype) {
        case DataType::TYPE_FP16: return tensorrt_llm::kernels::DATA_TYPE_FP16;
#ifdef ENABLE_BF16
        case DataType::TYPE_BF16: return tensorrt_llm::kernels::DATA_TYPE_BF16;
#endif
        default: throw std::runtime_error("not support dtype");
    }

}



bool cufmha::trtV1FmhaSupport() {
#ifdef USE_OLD_TRT_FMHA
    trtv1_fmha_runner_.reset(
    new FusedMHARunnerFP16v2(head_num_, size_per_head_, get_sm(), q_scaling_));

    return trtv1_fmha_runner_->fmha_supported(mtype_ == AttentionMaskType::causalMask) &&
        (head_num_ == kv_head_num_) && (dtype_ == DataType::TYPE_FP16);
#else
    return false;
#endif
}

void cufmha::runTrtV1Fmha(void* input,
                          void* cu_seqlens,
                          void* output,
                          void* qkv_buf_temp,
                          size_t batch_size,
                          size_t seq_len,
                          size_t token_num)
{
#ifdef USE_OLD_TRT_FMHA
    if (mtype_ == AttentionMaskType::causalMask) {
        trtv1_fmha_runner_->setup_causal_masked_fmha(seq_len, batch_size);
        trtv1_fmha_runner_->run_causal_masked_fmha(input,
                                                cu_seqlens,
                                                output,
                                                true,
                                                stream_);
    } else {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(dtype_, invokeTransposeAxis12,
                                        qkv_buf_temp,
                                        input,
                                        token_num,
                                        3,
                                        head_num_,
                                        size_per_head_,
                                        stream_);

        auto max_length  = trtv1_fmha_runner_->getSFromMaxSeqLen(seq_len);
        trtv1_fmha_runner_->setup(max_length, batch_size);
        trtv1_fmha_runner_->run(qkv_buf_temp,
                                nullptr,
                                cu_seqlens,
                                nullptr,
                                output,
                                stream_);
    }
#else
    return;
#endif

}


bool cufmha::trtV2FmhaSupport() {
    trtv2_fmha_runner_.reset(
        new tensorrt_llm::kernels::FusedMHARunnerV2(
            trtDtypeConvert(dtype_), head_num_, size_per_head_, q_scaling_));

    return trtv2_fmha_runner_->fmha_supported() &&
           (mtype_ == AttentionMaskType::causalMask ||
            mtype_ == AttentionMaskType::noMask) &&
        !(mtype_ == AttentionMaskType::noMask && use_linear_bias_slopes_);
}

bool cufmha::openSourceFmhaSupport()
{
    return (head_num_ % kv_head_num_ == 0) &&
           (mtype_ == AttentionMaskType::causalMask ||
            mtype_ == AttentionMaskType::noMask) &&
           ((size_per_head_ == 64) || (size_per_head_ == 96) || (size_per_head_ == 128));
}

void cufmha::runTrtV2FmhaPaged(void*  input,
                               void*  cu_q_seqlens,
                               void*  cu_kv_seqlens,
                               void*  output,
                               size_t batch_size,
                               size_t input_seq_len,
                               size_t max_past_kv_len,
                               size_t token_num,
                               KVBlockArray kv_block_array,
                               bool   mFMHAForceFP32Acc,
                               bool   mRemovePadding,
                               bool   is_alibi,
                               bool   is_alibi_with_sacle) {
    trtv2_fmha_runner_->setup_flags(mFMHAForceFP32Acc,
                                    mRemovePadding,
                                    (mtype_ == AttentionMaskType::causalMask),
                                    kv_head_num_);
    // By default, max_kv_cache_length == cyclic_kv_cache_length
    // unless each layer has different cyclic kv cache length.
    // Max cache capacity (used to allocate KV cache)
    // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
    trtv2_fmha_runner_->setup_paged_kv(batch_size,
                                       input_seq_len,
                                       max_past_kv_len,
                                       kv_block_array.mMaxBlocksPerSeq,
                                       kv_block_array.mTokensPerBlock,
                                       max_past_kv_len, //  cyclic_kv_cache_length
                                       token_num,
                                       is_alibi,
                                       is_alibi_with_sacle,
                                       1,
                                       0);

    trtv2_fmha_runner_->run_paged_kv(input,
                                     nullptr,
                                     nullptr,
                                     kv_block_array,
                                     cu_q_seqlens,
                                     cu_kv_seqlens,
                                     output,
                                     stream_);
    sync_check_cuda_error();
}

void cufmha::runTrtV2Fmha(void* input,
                          void* cu_seqlens,
                          void* output,
                          size_t batch_size,
                          size_t seq_len,
                          size_t token_num,
                          bool mFMHAForceFP32Acc,
                          bool mRemovePadding,
                          bool is_alibi,
                          bool is_alibi_with_sacle) {

    trtv2_fmha_runner_->setup_flags(mFMHAForceFP32Acc,
                                    mRemovePadding,
                                    (mtype_ == AttentionMaskType::causalMask),
                                    kv_head_num_);

    trtv2_fmha_runner_->setup(batch_size,
                              seq_len,
                              seq_len,
                              token_num,
                              is_alibi,
                              is_alibi_with_sacle,
                              1,
                              0);
    trtv2_fmha_runner_->run(input,
                            cu_seqlens,
                            output,
                            stream_);

    sync_check_cuda_error();
}

void cufmha::runOpenSourceFmhaPaged(void*  q,
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
                                    void* workspace,
                                    float* linear_bias_slopes)
{
   FT_CHECK_WITH_INFO(head_num_ % kv_head_num_ == 0, "Number of heads in key/value must divide number of heads in query");
   FT_CHECK_WITH_INFO(seq_size_per_block % 256 == 0, "open source fmha paged seq_size_per_block must be divided by 256");
    const auto seq_len_round = roundMultiple(seq_len, 32);

    FT_CHECK_WITH_INFO(block_table, "open source paged must have block_table");
    Flash_fwd_params flash_fwd_params = genFlashFwdParams(q, k, v, output, cu_seqlens, cu_kv_seqlens, workspace, batch_size, seq_len, seq_size_per_block * block_table_batch_stride, linear_bias_slopes);
    const int hidden_units_kv = kv_head_num_ * size_per_head_;

    // Set the pointers and strides.
    // All stride are in elements, not bytes.
    flash_fwd_params.k_row_stride  = size_per_head_;
    flash_fwd_params.v_row_stride  = size_per_head_;
    flash_fwd_params.k_head_stride = seq_size_per_block * size_per_head_;
    flash_fwd_params.v_head_stride = seq_size_per_block * size_per_head_;
    flash_fwd_params.k_batch_stride = seq_size_per_block * (hidden_units_kv);
    flash_fwd_params.v_batch_stride = seq_size_per_block * (hidden_units_kv);

    flash_fwd_params.block_table = block_table;
    flash_fwd_params.block_table_batch_stride = block_table_batch_stride;
    flash_fwd_params.page_block_size = seq_size_per_block;

    int device_id;
    int multi_processor_count = 1;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
    flash_fwd_params.num_splits = getNumSplits(batch_size, seq_len, seq_size_per_block * block_table_batch_stride);

    FT_CHECK_WITH_INFO(flash_fwd_params.num_splits <= 128, "open source not support split head 128");
    if (flash_fwd_params.num_splits > 1) {
        flash_fwd_params.softmax_lseaccum_ptr = (char*)workspace + sizeof(float) * batch_size * head_num_ * seq_len_round;
        flash_fwd_params.oaccum_ptr = (char*)(flash_fwd_params.softmax_lseaccum_ptr) + sizeof(float) * flash_fwd_params.num_splits * batch_size * head_num_ * seq_len_round;
    }
    run_mha_fwd(flash_fwd_params, stream_, block_table);
    sync_check_cuda_error();
}

void cufmha::runOpenSourceFmha(void* q,
                               void* k,
                               void* v,
                               void* output,
                               int* cu_seqlens,
                               size_t batch_size,
                               size_t seq_len,
                               void* workspace,
                               float* linear_bias_slopes)
{
    Flash_fwd_params flash_fwd_params = genFlashFwdParams(q, k, v, output, cu_seqlens, cu_seqlens, workspace, batch_size, seq_len, seq_len, linear_bias_slopes);
    run_mha_fwd(flash_fwd_params, stream_, false);
    sync_check_cuda_error();
}

Flash_fwd_params cufmha::genFlashFwdParams(void* q,
                                           void* k,
                                           void* v,
                                           void* output,
                                           int* cu_seqlens,
                                           int* cu_kv_seqlens,
                                           void* softmax_lse,
                                           size_t batch_size,
                                           size_t seq_len_q,
                                           size_t seq_len_kv,
                                           float* linear_bias_slopes) const
{
    const int head_size_rounded = roundMultiple(size_per_head_, 32);
    const int seqlen_q_rounded  = roundMultiple(seq_len_q, 128);
    const int seqlen_kv_rounded = roundMultiple(seq_len_kv, 128);
    Flash_fwd_params flash_fwd_params;
    memset(&flash_fwd_params, 0, sizeof(flash_fwd_params));
    flash_fwd_params.is_bf16 = (dtype_ == DataType::TYPE_BF16);
    const int hidden_units = head_num_ * size_per_head_;
    const int hidden_units_kv = kv_head_num_ * size_per_head_;
    flash_fwd_params.q_ptr = q;
    flash_fwd_params.k_ptr = k;
    flash_fwd_params.v_ptr = v;

    flash_fwd_params.q_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.k_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.v_row_stride  = hidden_units + 2 * hidden_units_kv;
    flash_fwd_params.q_head_stride = size_per_head_;
    flash_fwd_params.k_head_stride = size_per_head_;
    flash_fwd_params.v_head_stride = size_per_head_;
    flash_fwd_params.o_ptr         = output;
    flash_fwd_params.o_row_stride  = hidden_units;
    flash_fwd_params.o_head_stride = size_per_head_;

    flash_fwd_params.q_batch_stride = seq_len_q * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.k_batch_stride = seq_len_kv * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.v_batch_stride = seq_len_kv * (hidden_units + 2 * hidden_units_kv);
    flash_fwd_params.o_batch_stride = seq_len_q * hidden_units;

    flash_fwd_params.cu_seqlens_q = cu_seqlens;
    flash_fwd_params.cu_seqlens_k = cu_kv_seqlens;

    // P = softmax(QK^T)
    flash_fwd_params.p_ptr = nullptr;

    // Softmax sum
    flash_fwd_params.softmax_lse_ptr = softmax_lse;

    // Set the dimensions.
    flash_fwd_params.b                = batch_size;
    flash_fwd_params.h                = head_num_;
    flash_fwd_params.h_k              = kv_head_num_;
    flash_fwd_params.h_h_k_ratio      = head_num_ / kv_head_num_;
    flash_fwd_params.seqlen_q         = seq_len_q;
    flash_fwd_params.seqlen_k         = seq_len_kv;
    flash_fwd_params.seqlen_q_rounded = seqlen_q_rounded;
    flash_fwd_params.seqlen_k_rounded = seqlen_kv_rounded;
    flash_fwd_params.d                = size_per_head_;
    flash_fwd_params.d_rounded        = head_size_rounded;

    // Set the different scale values.
    float softmax_scale = (1.0f / sqrtf(size_per_head_ * 1.0f));
    flash_fwd_params.scale_softmax      = softmax_scale;
    flash_fwd_params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    float p_dropout             = 0.0f;
    flash_fwd_params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    flash_fwd_params.p_dropout_in_uint8_t     = uint8_t(std::floor(flash_fwd_params.p_dropout * 255.0));
    flash_fwd_params.rp_dropout               = 1.f / flash_fwd_params.p_dropout;
    flash_fwd_params.scale_softmax_rp_dropout = flash_fwd_params.rp_dropout * flash_fwd_params.scale_softmax;

    flash_fwd_params.is_causal = (mtype_ == AttentionMaskType::causalMask);
    if (linear_bias_slopes) {
        flash_fwd_params.alibi_slopes_ptr = linear_bias_slopes;
    }
    flash_fwd_params.is_seqlens_k_cumulative = true;
    return flash_fwd_params;
}

size_t cufmha::getOpenSourceWorkSpaceSize(size_t batch_size,
                                          size_t seqlen_q,
                                          size_t seqlen_k,
                                          bool   paged)
{
    size_t total_size = 0;
    const size_t seqlen_q_round = roundMultiple(seqlen_q, 32);
    total_size += sizeof(float) * batch_size * head_num_ * seqlen_q_round; // softmax_lse
    if (paged) {
        const size_t num_splits = getNumSplits(batch_size, seqlen_q, seqlen_k);
        if (num_splits > 1) {
            total_size += sizeof(float) * num_splits * batch_size * head_num_ * seqlen_q_round; // softmax_lseaccum
            total_size += sizeof(float) * num_splits * batch_size * head_num_ * roundMultiple(size_per_head_, 32) * seqlen_q_round; // oaccum
        }
    }
    return total_size;
}

int cufmha::getNumSplits(size_t batch_size,
                         size_t seqlen_q,
                         size_t seqlen_k) const
{
    int device_id;
    int multi_processor_count = 1;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
    const int block_n = size_per_head_ <= 64 ? 256 : (size_per_head_ <= 128 ? 128 : 64);
    const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
    const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
    return num_splits_heuristic(batch_size * head_num_ * num_m_blocks, multi_processor_count, num_n_blocks, 128);
}

}

#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention.h"
#include "src/fastertransformer/kernels/kv_cache_utils.h"
#include <iostream>


using namespace std;
namespace fastertransformer {

KVBlockArray getKVBlockArray(const AttentionModuleParams& params,
                             const Buffer& block_pointers,
                             const Buffer& block_scale_pointers,
                             int batch_size, cudaStream_t stream) {
    const auto& kv_cache = params.common.kv_cache;
    const auto& kv_blocks_offset = *(kv_cache->kv_cache_offset);
    RUNTIME_ASSERT_OP_ARG(
        kv_blocks_offset.shape()[0] == batch_size,
        "context attention kv blocks batch size expected [%d] but buffer[%s]",
        (int)batch_size, kv_blocks_offset.debugString().c_str());
    const auto max_blocks_per_batch = kv_blocks_offset.shape()[1];
    KVBlockArray kv_block_array(
        batch_size, max_blocks_per_batch, params.configs.tokens_per_block, 0);
    RUNTIME_ASSERT_OP_ARG(
            kv_cache->k_cache_buffer && kv_cache->v_cache_buffer,
            "kv cache buffer should has value when use kv_cache_offset");
    const auto& k_cache = *(kv_cache->k_cache_buffer);
    const auto& v_cache = *(kv_cache->v_cache_buffer);

    invokeConvertOffsetToAddrOneLayer(
            (uint64_t *)block_pointers.data(),
            (uint64_t)k_cache.data(),
            (uint64_t)v_cache.data(),
            (int *)kv_blocks_offset.data(),
            batch_size,
            max_blocks_per_batch,
            k_cache[0].sizeBytes(),
            stream);

    kv_block_array.data        = (int64_t *)block_pointers.data();
    if (kv_cache->k_scale_buffer) {
        RUNTIME_ASSERT_OP_ARG(
                kv_cache->v_scale_buffer,
                "v scale buffer should has value when use k scale buffer has value");
        const auto& k_scale = *(kv_cache->k_scale_buffer);
        const auto& v_scale = *(kv_cache->v_scale_buffer);

        invokeConvertOffsetToAddrOneLayer(
                (uint64_t*)block_scale_pointers.data(),
                (uint64_t)k_scale.data(),
                (uint64_t)v_scale.data(),
                (int*)kv_blocks_offset.data(),
                batch_size,
                max_blocks_per_batch,
                k_scale[0].sizeBytes(),
                stream);
        kv_block_array.scale     = (int64_t *)(block_scale_pointers.data());
        kv_block_array.int8_mode = true;
    }
    sync_check_cuda_error();
    return kv_block_array;
}

template <typename T>
void writeContextKvCache(
        const AttentionModuleParams& params,
        const Buffer& k, const Buffer& v,
        KVBlockArray kv_block_array,
        cudaStream_t stream)
{
    invokeTranspose4dBatchMajor<T, KVBlockArray>(
        k.data<T>(),
        v.data<T>(),
        kv_block_array,
        params.common.context_batch_size,
        params.common.context_max_seq_len + params.common.max_prefix_length,
        params.configs.size_per_head,
        params.configs.kv_head_num,
        params.common.kv_cache->k_scale_buffer ? KvCacheDataType::INT8: KvCacheDataType::BASE,
        nullptr,  // kvScaleOrigQuant
        params.common.input_lengths.dataWithOffset<int32_t>(params.common.decoder_batch_size),
        params.common.prefix_prompt_lengths ? params.common.prefix_prompt_lengths->data<int>() : nullptr,
        stream);
}

void MHA(const AttentionModuleParams& params,
         FMHAType                     fmha_type,
         cufmha*                      cufmha_runner,
         KVBlockArray                 kv_block_array,
         const BufferPtr&             q_output,
         const BufferPtr&             k_output,
         const BufferPtr&             v_output,
         cudaStream_t                 stream,
         DeviceBase*                  device) {
    FT_LOG_DEBUG("FMHA Type use %s.", std::to_string((int)fmha_type).c_str());
    auto datatype            = params.input.type();
    auto token_num           = params.input.shape()[0];
    auto batch_size          = params.common.context_batch_size;
    auto seq_len             = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto head_num            = params.configs.head_num;
    auto kv_head_num         = params.configs.kv_head_num;
    auto size_per_head       = params.configs.size_per_head;
    switch (fmha_type) {
        case FMHAType::PAGED_TRT_V2: {
            cufmha_runner->runTrtV2FmhaPaged(q_output->data(),
                                             params.common.cu_seqlens->data(),
                                             params.common.cu_kv_seqlens->data(),
                                             params.output.data(),
                                             batch_size,
                                             seq_len,
                                             seq_len_with_prefix,
                                             token_num,
                                             kv_block_array,
                                             false,
                                             false,
                                             params.common.linear_bias_slopes != nullptr,
                                             false);
            break;
        }
        case FMHAType::TRT_V2: {
            cufmha_runner->runTrtV2Fmha(params.input.data(),
                                        params.common.cu_seqlens->data(),
                                        params.output.data(),
                                        batch_size,
                                        seq_len,
                                        token_num,
                                        false,
                                        false,
                                        params.common.linear_bias_slopes != nullptr,
                                        false);
            break;
        }
        case FMHAType::PAGED_OPEN_SOURCE: {
            const size_t max_blocks_per_batch = params.common.kv_cache->kv_cache_offset->shape()[1];
            const auto   ws_size              = cufmha_runner->getOpenSourceWorkSpaceSize(
                batch_size, seq_len, max_blocks_per_batch * params.configs.tokens_per_block, true);
            auto ws = device->allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
                                             {"open_source_paged_fmha_ws"});
            cufmha_runner->runOpenSourceFmhaPaged(
                params.input.data(),
                params.common.kv_cache->k_cache_buffer->data(),
                params.common.kv_cache->v_cache_buffer->data(),
                params.output.data(),
                params.common.cu_seqlens->data<int>(),
                params.common.cu_kv_seqlens->data<int>(),
                params.common.kv_cache->kv_cache_offset->data<int>(),
                batch_size,
                max_blocks_per_batch,
                params.configs.tokens_per_block,
                seq_len,
                ws->data(),
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr);
            break;
        }
        case FMHAType::OPEN_SOURCE: {
            const auto   ws_size      = cufmha_runner->getOpenSourceWorkSpaceSize(batch_size, seq_len);
            auto         ws           = device->allocateBuffer({DataType::TYPE_INT8, {ws_size}, AllocationType::DEVICE},
                                                               {"open_source_fmha_ws"});
            const size_t hidden_units = head_num * size_per_head;
            const size_t hidden_units_kv = kv_head_num * size_per_head;
            cufmha_runner->runOpenSourceFmha(
                params.input.data(),
                params.input.dataWithOffset(hidden_units),
                params.input.dataWithOffset(hidden_units + hidden_units_kv),
                params.output.data(),
                params.common.cu_seqlens->data<int>(),
                batch_size,
                seq_len,
                ws->data(),
                params.common.linear_bias_slopes ? params.common.linear_bias_slopes->data<float>() : nullptr);
            break;
        }
        case FMHAType::TRT_V1: {
            auto qkv_buf_temp = device->allocateBuffer(
                {datatype, {token_num, head_num + 2 * kv_head_num, size_per_head}, AllocationType::DEVICE},
                {"qkv_buf_temp"});
            cufmha_runner->runTrtV1Fmha(params.input.data(),
                                        params.common.cu_seqlens->data(),
                                        params.output.data(),
                                        qkv_buf_temp->data(),
                                        batch_size,
                                        seq_len,
                                        token_num);
            break;
        }
        default: {
            q_output->updateShape({batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, size_per_head});
            auto qk_output = device->gemm({*q_output,
                                           *k_output,
                                           std::nullopt,
                                           nullptr,
                                           DataType::TYPE_FP32,
                                           TransposeOperation::NONE,
                                           TransposeOperation::TRANSPOSE});
            qk_output->updateShape({batch_size, head_num, seq_len, seq_len_with_prefix});
            printBufferData(*qk_output, "qk_output: ");
            float scale = (1.0f / sqrtf(size_per_head * 1.0f));
            // TODO(lidongjin): Only support float32(in)\float16(output).
            auto softmax_type = qk_output->type();
            RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                                  "attention_mask must be provided for default context attention implementation");
            auto softmax_qk_output = device->softmax({std::move(qk_output),
                                                      *params.common.attention_mask,
                                                      std::nullopt,
                                                      scale,
                                                      datatype,
                                                      params.common.linear_bias_slopes ?
                                                          (OptionalConstBufferRef)*params.common.linear_bias_slopes :
                                                          std::nullopt});
            softmax_qk_output->updateShape(
                {batch_size, kv_head_num, (head_num / kv_head_num) * seq_len, seq_len_with_prefix});
            printBufferData(*softmax_qk_output, "softmax_qk_output: ");
            auto qkv_output = device->gemm({*softmax_qk_output, *v_output});
            qkv_output->updateShape({batch_size, head_num, seq_len, size_per_head});
            printBufferData(*qkv_output, "qkv_output");
            auto& qkv_transpose_output = params.output;
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                             invokeTransposeAttentionOutRemovePadding,
                                             qkv_output->data(),
                                             qkv_transpose_output.data(),
                                             token_num,
                                             batch_size,
                                             seq_len,
                                             head_num,
                                             size_per_head,
                                             params.common.padding_offset->data<int>(),
                                             nullptr,
                                             0,
                                             stream);
        }
    }
}

AttentionModuleOutput CudaDevice::contextAttention(const AttentionModuleParams& params) {
    auto datatype       = params.input.type();
    auto token_num      = params.input.shape()[0];
    auto batch_size     = params.common.context_batch_size;
    auto decoder_batch_size = params.common.decoder_batch_size;
    auto seq_len        = params.common.context_max_seq_len;
    auto seq_len_with_prefix = seq_len + params.common.max_prefix_length;
    auto head_num       = params.configs.head_num;
    auto kv_head_num    = params.configs.kv_head_num;
    auto size_per_head  = params.configs.size_per_head;

    auto q_output = allocateBuffer({params.input.type(),
                                    {batch_size, head_num, seq_len, size_per_head},
                                    AllocationType::DEVICE},
                                    {"q_output"});

    auto k_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len_with_prefix, size_per_head},
                                    AllocationType::DEVICE},
                                    {"k_output"});

    auto v_output = allocateBuffer({params.input.type(),
                                    {batch_size, kv_head_num, seq_len_with_prefix, size_per_head},
                                    AllocationType::DEVICE},
                                    {"v_output"});

    // allocate qkv should be better
    if (fmha_type_ == FMHAType::NONE) {
        cudaMemsetAsync(q_output->data(), 0, q_output->sizeBytes(), stream_);
        cudaMemsetAsync(k_output->data(), 0, k_output->sizeBytes(), stream_);
        cudaMemsetAsync(v_output->data(), 0, v_output->sizeBytes(), stream_);
    }

    KVBlockArray kv_block_array;
    BufferPtr block_pointers, block_scale_pointers;
    PrefixPromptBatchWeightsParam prefix_prompt_param;
    if (params.common.kv_cache) {
        const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_offset->shape()[1];
        block_pointers = allocateBuffer({DataType::TYPE_INT64,
                                         {batch_size, 1, 2, max_blocks_per_batch},
                                         AllocationType::DEVICE},
            {"kv_block_pointers"});
        block_scale_pointers = allocateBuffer({DataType::TYPE_INT64,
                                               {batch_size, 1, 2, max_blocks_per_batch},
                                               AllocationType::DEVICE},
            {"kv_scale_pointers"});
        kv_block_array = getKVBlockArray(params, *block_pointers, *block_scale_pointers, batch_size, stream_);

        if (params.common.prefix_prompt_lengths) {
            prefix_prompt_param.d_prefix_prompt_lengths = params.common.prefix_prompt_lengths->data<int>();
            prefix_prompt_param.max_prefix_prompt_length = params.common.max_prefix_length;
            prefix_prompt_param.count_length = 1;
            prefix_prompt_param.kv_block_array = kv_block_array;
        }
    }

    // int8
    float*  scale_out_ptr    = nullptr;
    int     int8_mode        = 0;

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype, invokeAddFusedQKVBiasTranspose,
        q_output->data(),
        k_output->data(),
        v_output->data(),
        &prefix_prompt_param,
        params.input.data(),
        params.common.position_ids ? params.common.position_ids->dataWithOffset<int>(decoder_batch_size): nullptr,
        params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias ? params.weights.qkv_weight->bias->data() : nullptr,
        params.common.padding_offset->data<int>(),
        params.common.cu_seqlens->data<int>(),
        batch_size,
        seq_len,
        token_num,
        head_num,
        kv_head_num,
        size_per_head,
        params.configs.rope_config,
        params.configs.use_logn_attn,
        scale_out_ptr,
        int8_mode,
        fmha_type_ == FMHAType::PAGED_TRT_V2,
        stream_
    );
    sync_check_cuda_error();

    printBufferData(params.input, "after invoke transpse");

    if (params.common.kv_cache) {
        DISPATCH_CUDA_FUNCTION_DATA_TYPE(
                datatype, writeContextKvCache,
                std::cref(params),
                std::cref(*k_output),
                std::cref(*v_output),
                kv_block_array,
                stream_);
        sync_check_cuda_error();
    }
    MHA(params, fmha_type_, cufmha_runner_.get(), kv_block_array, q_output, k_output, v_output, stream_, this);
}

template<typename T>
void selfAttentionwrapper(const AttentionModuleParams params,
                          bool use_multi_block_mode,
                          size_t max_seq_len_tile,
                          void* partial_out,
                          float* partial_sum,
                          float* partial_max,
                          int* block_counter,
                          KVBlockArray kv_block_array,
                          cudaStream_t stream)
{
    size_t token_num            = params.input.shape()[0];
    size_t batch_size           = params.common.decoder_batch_size;
    size_t step                 = params.common.decoder_max_seq_len + 1;
    size_t local_head_num       = params.configs.head_num;
    size_t local_head_num_kv    = params.configs.kv_head_num;
    size_t size_per_head        = params.configs.size_per_head;
    const auto& output = params.output;

    const T* qkv_buf_ptr = params.input.data<T>();
    T* qkv_buf_2_ = output.data<T>();

    const T* bias_ptr = (params.weights.qkv_weight->bias == nullptr || !params.configs.fuse_qkv_add_bias) ?
                         nullptr :
                         params.weights.qkv_weight->bias->data<T>();

    // TODO(lidongjin) support relative attention
    const T* relative_attention_bias_ptr = nullptr;

    // prefix prompt

    auto prefix_lengths = params.common.prefix_prompt_lengths ? params.common.prefix_prompt_lengths->data<int>() : nullptr;

    const auto* input_lengths = params.common.input_lengths.data<int>();
    const auto* sequence_lengths = params.common.sequence_lengths.data<int>();

    float q_scaling = params.configs.q_scaling;
    int relative_attention_bias_stride = 0;
    const float* linear_bias_slopes = nullptr;
    const bool* masked_tokens = nullptr;

    // TODO(lidongjin) support int8
    const float* query_weight_scale_out = nullptr;
    const float* attention_output_weight_scale_out = nullptr;
    int int8_mode = 0;

    fusedQKV_masked_attention_dispatch<T, KVBlockArray>(
        qkv_buf_ptr,
        bias_ptr,
        relative_attention_bias_ptr,
        nullptr, // cache_indir
        qkv_buf_2_,
        nullptr, // finished
        sequence_lengths,
        batch_size,
        1, // beam_width
        local_head_num,
        local_head_num_kv,
        size_per_head,
        params.configs.rope_config,
        params.configs.use_logn_attn,
        params.common.position_ids ? params.common.position_ids->data<int>() : nullptr,
        step,
        nullptr, // prefix_prompt_lengths
        0, // max_prefix_prompt_length
        true, // count_prefix_length
        input_lengths,
        step,
        q_scaling,
        relative_attention_bias_stride,
        linear_bias_slopes,
        masked_tokens,
        query_weight_scale_out,
        attention_output_weight_scale_out,
        int8_mode,
        use_multi_block_mode,
        (int)max_seq_len_tile,
        reinterpret_cast<T*>(partial_out),
        partial_sum,
        partial_max,
        block_counter,
        kv_block_array,
        stream);

    sync_check_cuda_error();
}

AttentionModuleOutput CudaDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    auto datatype = params.input.type();
    size_t max_seq_len_tile = 0;
    BufferPtr partial_out = nullptr;
    BufferPtr partial_sum = nullptr;
    BufferPtr partial_max = nullptr;
    BufferPtr block_counter = nullptr;

    size_t batch_size           = params.common.decoder_batch_size;
    size_t local_head_num       = params.configs.head_num;
    size_t size_per_head        = params.configs.size_per_head;

    if (use_multi_block_mode) {
        const int threads_per_value = pow2roundup(size_per_head) * getTypeSize(datatype) / 16;
        // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
        max_seq_len_tile = 256 / threads_per_value;
        partial_out = allocateBuffer({datatype,
                                     {batch_size, max_seq_len_tile, local_head_num, size_per_head},
                                     AllocationType::DEVICE},
                                     {"partial_out"});
        partial_sum = allocateBuffer({DataType::TYPE_FP32,
                                     {batch_size, max_seq_len_tile, local_head_num},
                                     AllocationType::DEVICE},
                                     {"partial_sum"});
        partial_max = allocateBuffer({DataType::TYPE_FP32,
                                     {batch_size, max_seq_len_tile, local_head_num},
                                     AllocationType::DEVICE},
                                     {"partial_max"});
        block_counter = allocateBuffer({DataType::TYPE_INT32,
                                      {batch_size, local_head_num},
                                      AllocationType::DEVICE},
                                      {"block_counter"});
        // TODO(lidongjin) use fill op to set zeros.
        cudaMemsetAsync(block_counter->data(), 0, sizeof(int) * batch_size * local_head_num, stream_);
    }
    void* partial_out_data = (partial_out == nullptr) ? nullptr : partial_out->data();
    float* partial_sum_data = (partial_sum == nullptr) ? nullptr : partial_sum->data<float>();
    float* partial_max_data = (partial_max == nullptr) ? nullptr : partial_max->data<float>();
    int* block_counter_data = (block_counter == nullptr) ? nullptr : block_counter->data<int>();

    RUNTIME_ASSERT_OP_ARG(
        params.common.kv_cache, "kv cache can not be null for decoder self-attention");
    const auto max_blocks_per_batch = params.common.kv_cache->kv_cache_offset->shape()[1];
    auto block_pointers = allocateBuffer({DataType::TYPE_INT64,
                                          {batch_size, 1, 2, max_blocks_per_batch},
                                          AllocationType::DEVICE},
        {"kv_block_pointers"});
    auto block_scale_pointers = allocateBuffer({DataType::TYPE_INT64,
                                                {batch_size, 1, 2, max_blocks_per_batch},
                                                AllocationType::DEVICE},
        {"kv_scale_pointers"});
    KVBlockArray kv_block_array = getKVBlockArray(params, *block_pointers, *block_scale_pointers, batch_size, stream_);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(datatype,
                                     selfAttentionwrapper,
                                     params,
                                     use_multi_block_mode,
                                     max_seq_len_tile,
                                     partial_out_data,
                                     partial_sum_data,
                                     partial_max_data,
                                     block_counter_data,
                                     kv_block_array,
                                     stream_);
}

} // namespace fastertransformer

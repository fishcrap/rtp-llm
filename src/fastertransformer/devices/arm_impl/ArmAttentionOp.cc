#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>
#include <openblas/cblas.h>

namespace fastertransformer {

/* Input has shape [dim0, dim1, dim2, dim3] */
void transposeDim12(BufferPtr input, Buffer& output) {
    auto dim     = input->shape();
    auto elem_sz = input->typeSize();

    for (int k = 0; k < dim[0]; k++) {
        for (int i = 0; i < dim[2]; i++) {
            for (int j = 0; j < dim[1]; j++) {
                memcpy(output.dataWithOffset(k * dim[1] * dim[2] * dim[3] + (i * dim[1] + j) * dim[3]),
                       input->dataWithOffset(k * dim[1] * dim[2] * dim[3] + (j * dim[2] + i) * dim[3]),
                       elem_sz * dim[3]);
            }
        }
    }
}

void getCacheAddrFromIndex(const KvCacheInfo& kv_cache, size_t batch, size_t block_idx, void **k_addr, void **v_addr) {
    const auto& kv_blocks_offset = *(kv_cache.kv_cache_offset);
    const auto& k_cache = *(kv_cache.k_cache_buffer);
    const auto& v_cache = *(kv_cache.v_cache_buffer);
    const auto  max_blocks_per_batch = kv_blocks_offset.shape()[1];
    size_t block_size = k_cache[0].sizeBytes();
    int    *index = (int *)kv_blocks_offset.data();

    *k_addr = (char*)k_cache.data() + index[batch * max_blocks_per_batch + block_idx] * block_size;
    *v_addr = (char*)v_cache.data() + index[batch * max_blocks_per_batch + block_idx] * block_size;
}

void assemCache(const AttentionModuleParams& params, BufferPtr k_out, BufferPtr v_out, size_t tokens_per_block) {
    auto   elem_sz          = k_out->typeSize();
    auto   batch_size       = k_out->shape()[0];
    auto   head_num         = k_out->shape()[1];
    auto   kv_seq_len       = k_out->shape()[2];
    auto   head_dim         = k_out->shape()[3];
    size_t blocks_per_batch = (kv_seq_len + tokens_per_block - 1) / tokens_per_block;
    size_t copied_len;

    void *k_block_addr;
    void *v_block_addr;
    for (int batch = 0; batch < batch_size; batch++) {
        copied_len = 0;
        for (int i = 0; i < blocks_per_batch; i++) {
            size_t len = std::min(tokens_per_block, kv_seq_len - copied_len);
            getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

            memcpy(k_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   k_block_addr,
                   elem_sz * len * head_num * head_dim);
            memcpy(v_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   v_block_addr,
                   elem_sz * len * head_num * head_dim);
            copied_len += len;
        }
    }
}

/* 'input' with shape [batch, seq_len, num_heads, head_dim]
 * 'bias' has shape [num_heads * head_dim]
 */
template<typename T>
void addQKVBias(Buffer& input, const void* bias) {
    size_t batch_sz  = input.shape()[0];
    size_t seq_len   = input.shape()[1];
    size_t num_heads = input.shape()[2];
    size_t head_dim  = input.shape()[3];
#if 1
    for (int i = 0; i < batch_sz; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < num_heads * head_dim; k++) {
                *(T*)(input.dataWithOffset((i * seq_len + j) * num_heads * head_dim + k)) += *((T*)bias + k);
            }
        }
    }
#else
    const int N = batch_sz * seq_len;
    const int hidden_size = num_heads * head_dim;
    parallel_for(N, [&](int tid) {
        for (int k = 0; k < hidden_size; k++) {
            *(float*)(input.dataWithOffset(tid * hidden_size + k)) += *((float*)bias + k);
        }
    });
#endif
}

void writeContextKVCache(const AttentionModuleParams& params, BufferPtr k, BufferPtr v) {
    const auto  batch_size           = params.common.context_batch_size;
    const auto  seq_len              = params.common.context_max_seq_len;

    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    auto block_tokens  = params.configs.tokens_per_block;

    size_t block_num = (seq_len + block_tokens - 1) / block_tokens;
    auto   elem_sz   = params.input.typeSize();
    size_t copied_len;

    void *k_block_addr;
    void *v_block_addr;
    for (int batch = 0; batch < batch_size; batch++) {
        copied_len = 0;
        for (int i = 0; i < block_num; i++) {
            size_t   len          = std::min(block_tokens, seq_len - copied_len);
            getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

            memcpy(k_block_addr,
                   k->dataWithOffset((batch * seq_len + i * block_tokens) * kv_head_num * size_per_head),
                   elem_sz * len * kv_head_num * size_per_head);
            memcpy(v_block_addr,
                   v->dataWithOffset((batch * seq_len + i * block_tokens) * kv_head_num * size_per_head),
                   elem_sz * len * kv_head_num * size_per_head);

            copied_len += len;
        }
    }
}

/* In decoding phase, 1 token for auto-regression each time. */
void updateKVCache(const AttentionModuleParams& params, BufferPtr k, BufferPtr v) {
    const auto  batch_size           = params.common.decoder_batch_size;
    int* decoder_len = static_cast<int*>(params.common.sequence_lengths.dataWithOffset(0));

    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    auto block_tokens  = params.configs.tokens_per_block;

    auto elem_sz = params.input.typeSize();

    void *k_block_addr;
    void *v_block_addr;
    for (int batch = 0; batch < batch_size; batch++) {
        int      step         = *static_cast<int*>(params.common.sequence_lengths.dataWithOffset(batch));
        size_t   block_offset = step / block_tokens;
        getCacheAddrFromIndex(params.common.kv_cache.value(), batch, block_offset, &k_block_addr, &v_block_addr);

        memcpy((uint8_t*)k_block_addr + (decoder_len[batch] % block_tokens) * kv_head_num * size_per_head * elem_sz,
               k->dataWithOffset(batch * kv_head_num * size_per_head),
               elem_sz * kv_head_num * size_per_head);
        memcpy((uint8_t*)v_block_addr + (decoder_len[batch] % block_tokens) * kv_head_num * size_per_head * elem_sz,
               v->dataWithOffset(batch * kv_head_num * size_per_head),
               elem_sz * kv_head_num * size_per_head);
    }
}

/* 'input' with shape [batch, seq_len, num_heads, head_dim] */
template<typename T>
void context_rope(Buffer& input, size_t base, size_t embed_dim) {
    size_t batch     = input.shape()[0];
    size_t seq_len   = input.shape()[1];
    size_t num_heads = input.shape()[2];
    size_t head_dim  = input.shape()[3];
    auto   type_size = input.typeSize();

    if ((type_size != 4) && (type_size != 2)) {
        throw std::runtime_error("RoPE input type is not supported");
    }

    size_t inv_freq_size = (embed_dim + 1) / 2;

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < num_heads; k++) {
                for (int d = 0; d < inv_freq_size; d++) {
                    float freq = 1.0f / powf(base, (float)(d * 2) / embed_dim);
                    float val  = freq * j;
                    float fcr  = cosf(val);
                    float fci  = sinf(val);
                    auto  v0   = input.dataWithOffset(((i * seq_len + j) * num_heads + k) * head_dim + d);
                    auto  v1 = input.dataWithOffset(((i * seq_len + j) * num_heads + k) * head_dim + d + inv_freq_size);
                    auto  d0 = *(T*)v0;
                    auto  d1 = *(T*)v1;
                    *(T*)v0  = d0 * fcr - d1 * fci;
                    *(T*)v1  = d0 * fci + d1 * fcr;
                }
            }
        }
    }
}

/* At auto-regression stage, seq_len is always 1 */
template<typename T>
void attention_rope(Buffer& input, size_t timestep, size_t base, size_t embed_dim) {
    size_t batch = input.shape()[0];
    // size_t seq_len = input.shape()[1];
    size_t num_heads = input.shape()[2];
    size_t head_dim  = input.shape()[3];
    auto   type_size = input.typeSize();

    if ((type_size != 4) && (type_size != 2)) {
        throw std::runtime_error("RoPE input type is not supported");
    }

    size_t inv_freq_size = (embed_dim + 1) / 2;
    for (int i = 0; i < batch; i++) {
        for (int k = 0; k < num_heads; k++) {
            for (int d = 0; d < inv_freq_size; d++) {
                float freq = 1.0f / powf(base, (float)(d * 2) / embed_dim);
                float val  = freq * timestep;
                float fcr  = cosf(val);
                float fci  = sinf(val);
                auto  v0   = input.dataWithOffset((i * num_heads + k) * head_dim + d);
                auto  v1   = input.dataWithOffset((i * num_heads + k) * head_dim + d + inv_freq_size);
                auto  d0   = *(T*)v0;
                auto  d1   = *(T*)v1;
                *(T*)v0    = d0 * fcr - d1 * fci;
                *(T*)v1    = d0 * fci + d1 * fcr;
            }
        }
    }
}

void ArmCpuDevice::printStat() {
    for (int i = 0; i < sizeof(a_cnt_) / sizeof(uint64_t); i++) {
        std::cout << "$$$   [" << i << "] time (us) - min: " << a_tmin_[i] << " ; max: " << a_tmax_[i] << " ; ave: " << a_tave_[i] / a_cnt_[i] << std::endl;
    }
}

void ArmCpuDevice::logTime(std::chrono::microseconds diff, size_t index) {
    if (diff.count() < a_tmin_[index])
        a_tmin_[index] = diff.count();
    if (diff.count() > a_tmax_[index])
        a_tmax_[index] = diff.count();
    a_tave_[index] += diff.count();
    a_cnt_[index] += 1;
}

#define USING_THREADED_MHA 0
void ArmCpuDevice::contextAttentionFallback(const AttentionModuleParams& params) {
    auto datatype      = params.input.type();
    auto batch_size    = params.common.context_batch_size;
    auto seq_len       = params.common.context_max_seq_len;
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    std::chrono::steady_clock::time_point tStart, tEnd;
    std::chrono::microseconds diff;

    tStart = std::chrono::steady_clock::now();
    // qkv to q_output, k_output, v_output
    auto qkv = params.input.data();

    printBufferData(params.input, "qkv");
    arm_compute::DataType acl_data_type = getAclDataType(datatype);

    arm_compute::NESplit    split;
    arm_compute::Tensor     src;
    arm_compute::TensorInfo src_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num + 2 * kv_head_num, seq_len, batch_size), 1, acl_data_type);
    ;
    src.allocator()->init(src_info);
    src.allocator()->import_memory(qkv);
    std::vector<arm_compute::Tensor>   dsts{};
    std::vector<arm_compute::ITensor*> dsts_ptr;

    arm_compute::TensorInfo q_info, kv_info;
    arm_compute::Tensor     q, k, v;
    q_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num, seq_len, batch_size), 1, acl_data_type);
    kv_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, seq_len, batch_size), 1, acl_data_type);

    auto q_input =
        allocateBuffer({params.input.type(), {batch_size, seq_len, head_num, size_per_head}, AllocationType::HOST}, {});

    auto k_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});

    auto v_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    q.allocator()->init(q_info);
    k.allocator()->init(kv_info);
    v.allocator()->init(kv_info);
    q.allocator()->import_memory(q_input->data());
    dsts.push_back(std::move(q));

    k.allocator()->import_memory(k_input->data());
    dsts.push_back(std::move(k));

    v.allocator()->import_memory(v_input->data());
    dsts.push_back(std::move(v));

    for (auto& dst : dsts) {
        dsts_ptr.emplace_back(&dst);
    }
    split.configure(&src, dsts_ptr, 1);
    split.run();
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 0);

    printBufferData(*q_input, "q_input");
    printBufferData(*k_input, "k_input");
    printBufferData(*v_input, "v_input");
    tStart = std::chrono::steady_clock::now();
    if (params.weights.qkv_weight->bias) {
        if (datatype == DataType::TYPE_FP32) {
            addQKVBias<float>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<float>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<float>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else if (datatype == DataType::TYPE_FP16) {
            addQKVBias<__fp16>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<__fp16>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<__fp16>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 1);

    tStart = std::chrono::steady_clock::now();
    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {
            if (datatype == DataType::TYPE_FP32) {
                context_rope<float>(*q_input,
                                    params.configs.rope_config.base,
                                    params.configs.rope_config.dim);
                context_rope<float>(*k_input,
                                    params.configs.rope_config.base,
                                    params.configs.rope_config.dim);
            } else if (datatype == DataType::TYPE_FP16) {
                context_rope<__fp16>(*q_input,
                                     params.configs.rope_config.base,
                                     params.configs.rope_config.dim);
                context_rope<__fp16>(*k_input,
                                     params.configs.rope_config.base,
                                     params.configs.rope_config.dim);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
        } else {
            throw std::runtime_error(fmtstr("SelfAttention RoPE type is not supported %d", params.configs.rope_config.style));
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 2);

    printBufferData(*q_input, "q_input pre-trans12");
    printBufferData(*k_input, "k_input pre-trans12");
    printBufferData(*v_input, "v_input pre-trans12");

    tStart = std::chrono::steady_clock::now();
    auto q_output =
        allocateBuffer({params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    auto k_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    auto v_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});
    transposeDim12(std::move(q_input), *q_output);
    transposeDim12(std::move(k_input), *k_output);
    transposeDim12(std::move(v_input), *v_output);
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 3);

    printBufferData(*q_output, "q_output");
    printBufferData(*k_output, "k_output");
    printBufferData(*v_output, "v_output");

    tStart = std::chrono::steady_clock::now();
    if (params.common.kv_cache) {
        writeContextKVCache(params, k_output, v_output);
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 8);

    tStart = std::chrono::steady_clock::now();
#if USING_THREADED_MHA
    auto qk_output = allocateBuffer({TYPE_FP32, {batch_size, head_num, seq_len, seq_len}, AllocationType::HOST}, {"qk_output"});
    const int N = batch_size * head_num;
    parallel_for(N, [&](int tid) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len, seq_len, size_per_head, 1.0,
            (const float*)q_output->dataWithOffset(tid * seq_len * size_per_head), size_per_head,
            (const float*)k_output->dataWithOffset(tid * seq_len * size_per_head), size_per_head, 0.0,
            (float*)qk_output->dataWithOffset(tid * seq_len * seq_len), seq_len);
    });
#else
    auto qk_output = gemm({*q_output,
                           *k_output,
                           std::nullopt,
                           nullptr,
                           DataType::TYPE_INVALID,
                           TransposeOperation::NONE,
                           TransposeOperation::TRANSPOSE});
#endif
    printBufferData(*qk_output, "qk_output: ");
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 4);

    tStart = std::chrono::steady_clock::now();
    float scale = (1.0f / sqrtf(size_per_head * 1.0f));

    RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                          "attention_mask must be provided for default context attention implementation");
    printBufferData(*params.common.attention_mask, "attention_mask: ");
    auto softmax_qk_output = softmax({std::move(qk_output), *params.common.attention_mask, std::nullopt, scale});
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 5);

    printBufferData(*softmax_qk_output, "softmax_qk_output: ");
    tStart = std::chrono::steady_clock::now();
#if USING_THREADED_MHA
    auto qkv_output = allocateBuffer({TYPE_FP32, {batch_size, head_num, seq_len, size_per_head}, AllocationType::HOST}, {"qkv_output"});
    const int NN = batch_size * head_num;
    parallel_for(NN, [&](int tid) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_len, size_per_head, seq_len, 1.0,
            (const float*)softmax_qk_output->dataWithOffset(tid * seq_len * seq_len), seq_len,
            (const float*)v_output->dataWithOffset(tid * seq_len * size_per_head), size_per_head, 0.0,
            (float*)qkv_output->dataWithOffset(tid * seq_len * size_per_head), size_per_head);
    });
#else
    auto qkv_output = gemm({*softmax_qk_output, *v_output});
#endif
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 6);

    tStart = std::chrono::steady_clock::now();
    transposeDim12(qkv_output, params.output);
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 7);

    // if (a_cnt_[0] == 24)
    //     printStat();
    printBufferData(*qkv_output, "qkv_output: ");
    printBufferData(params.output, "params.output: ");
}

void ArmCpuDevice::decoderSelfAttentionFallback(const AttentionModuleParams& params) {
    auto   datatype      = params.input.type();
    auto   batch_size    = params.common.decoder_batch_size;
    size_t kv_seq_len    = *static_cast<int*>(params.common.input_lengths.dataWithOffset(0));
    size_t seq_len       = *static_cast<int*>(params.common.sequence_lengths.dataWithOffset(0)) + 1 - kv_seq_len;
    auto   head_num      = params.configs.head_num;
    auto   kv_head_num   = params.configs.kv_head_num;
    auto   size_per_head = params.configs.size_per_head;
    auto   block_tokens  = params.configs.tokens_per_block;

    if (!params.common.kv_cache.has_value()) {
        throw std::runtime_error("kv cache block pointers can not be null");
    }

    // qkv to q_output, k_output, v_output
    auto qkv = params.input.data();

    printBufferData(params.input, "decoder qkv");
    arm_compute::DataType acl_data_type = getAclDataType(datatype);

    arm_compute::NESplit    split;
    arm_compute::Tensor     src;
    arm_compute::TensorInfo src_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num + 2 * kv_head_num, seq_len, batch_size), 1, acl_data_type);
    ;
    src.allocator()->init(src_info);
    src.allocator()->import_memory(qkv);
    std::vector<arm_compute::Tensor>   dsts{};
    std::vector<arm_compute::ITensor*> dsts_ptr;

    arm_compute::TensorInfo q_info, kv_info;
    arm_compute::Tensor     q, k, v;
    q_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num, seq_len, batch_size), 1, acl_data_type);
    kv_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, seq_len, batch_size), 1, acl_data_type);

    auto q_input =
        allocateBuffer({params.input.type(), {batch_size, seq_len, head_num, size_per_head}, AllocationType::HOST}, {});

    auto k_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});

    auto v_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    q.allocator()->init(q_info);
    k.allocator()->init(kv_info);
    v.allocator()->init(kv_info);
    q.allocator()->import_memory(q_input->data());
    dsts.push_back(std::move(q));

    k.allocator()->import_memory(k_input->data());
    dsts.push_back(std::move(k));

    v.allocator()->import_memory(v_input->data());
    dsts.push_back(std::move(v));

    for (auto& dst : dsts) {
        dsts_ptr.emplace_back(&dst);
    }
    split.configure(&src, dsts_ptr, 1);
    split.run();

    printBufferData(*q_input, "q_input");
    printBufferData(*k_input, "k_input");
    printBufferData(*v_input, "v_input");

    if (params.weights.qkv_weight->bias) {
        if (datatype == DataType::TYPE_FP32) {
            addQKVBias<float>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<float>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<float>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else if (datatype == DataType::TYPE_FP16) {
            addQKVBias<__fp16>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<__fp16>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<__fp16>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }

    printBufferData(*q_input, "biased q_input");
    printBufferData(*k_input, "biased k_input");
    printBufferData(*v_input, "biased v_input");

    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {

            if (datatype == DataType::TYPE_FP32) {
                attention_rope<float>(*q_input,
                                      seq_len + kv_seq_len - 1,
                                      params.configs.rope_config.base,
                                      params.configs.rope_config.dim);
                attention_rope<float>(*k_input,
                                      seq_len + kv_seq_len - 1,
                                      params.configs.rope_config.base,
                                      params.configs.rope_config.dim);
            } else if (datatype == DataType::TYPE_FP16) {
                attention_rope<__fp16>(*q_input,
                                       seq_len + kv_seq_len - 1,
                                       params.configs.rope_config.base,
                                       params.configs.rope_config.dim);
                attention_rope<__fp16>(*k_input,
                                       seq_len + kv_seq_len - 1,
                                       params.configs.rope_config.base,
                                       params.configs.rope_config.dim);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
        } else {
            throw std::runtime_error("SelfAttention RoPE type is not supported");
        }
    }

    auto q_output =
        allocateBuffer({params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    auto k_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    auto v_output = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len, size_per_head}, AllocationType::HOST}, {});
    transposeDim12(std::move(q_input), *q_output);
    transposeDim12(std::move(k_input), *k_output);
    transposeDim12(std::move(v_input), *v_output);

    printBufferData(*q_output, "q_output");
    printBufferData(*k_output, "k_output");
    printBufferData(*v_output, "v_output");

    updateKVCache(params, k_output, v_output);

    /* Retrieve k/v cache data and attach it to k/v tensor. */
    auto k_cache =
        allocateBuffer({datatype, {batch_size, kv_head_num, kv_seq_len, size_per_head}, AllocationType::HOST}, {});
    auto v_cache =
        allocateBuffer({datatype, {batch_size, kv_head_num, kv_seq_len, size_per_head}, AllocationType::HOST}, {});
    assemCache(params, k_cache, v_cache, block_tokens);

    printBufferData(*k_cache, "k_cache");
    printBufferData(*v_cache, "v_cache");
    auto k_with_cache = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len + kv_seq_len, size_per_head}, AllocationType::HOST},
        {});
    auto v_with_cache = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len + kv_seq_len, size_per_head}, AllocationType::HOST},
        {});
    arm_compute::NEConcatenateLayer          cat;
    std::vector<arm_compute::Tensor>         kv_inputs{};
    std::vector<const arm_compute::ITensor*> kv_inputs_vector;
    arm_compute::Tensor                      kv_cache, kv_cat;
    arm_compute::TensorInfo                  kv_cache_info, kv_cat_info;
    kv_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, seq_len, kv_head_num, batch_size), 1, acl_data_type);
    kv_cache_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_seq_len, kv_head_num, batch_size), 1, acl_data_type);
    kv_cat_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_seq_len + seq_len, kv_head_num, batch_size), 1, acl_data_type);
    kv_cache.allocator()->init(kv_cache_info);
    kv_cache.allocator()->import_memory(k_cache->data());
    kv_inputs.push_back(std::move(kv_cache));
    k.allocator()->init(kv_info);
    k.allocator()->import_memory(k_output->data());
    kv_inputs.push_back(std::move(k));
    for (auto& input : kv_inputs) {
        kv_inputs_vector.emplace_back(&input);
    }
    kv_cat.allocator()->init(kv_cat_info);
    kv_cat.allocator()->import_memory(k_with_cache->data());
    cat.configure(kv_inputs_vector, &kv_cat, 1);
    cat.run();

    kv_inputs.clear();
    kv_inputs_vector.clear();

    kv_cache.allocator()->init(kv_cache_info);
    kv_cache.allocator()->import_memory(v_cache->data());
    kv_inputs.push_back(std::move(kv_cache));
    v.allocator()->init(kv_info);
    v.allocator()->import_memory(v_output->data());
    kv_inputs.push_back(std::move(v));
    for (auto& input : kv_inputs) {
        kv_inputs_vector.emplace_back(&input);
    }
    kv_cat.allocator()->init(kv_cat_info);
    kv_cat.allocator()->import_memory(v_with_cache->data());
    cat.configure(kv_inputs_vector, &kv_cat, 1);
    cat.run();
    printBufferData(*k_with_cache, "k_with_cache");
    printBufferData(*v_with_cache, "v_with_cache");

    auto qk_output = gemm({*q_output,
                           *k_with_cache,
                           std::nullopt,
                           nullptr,
                           DataType::TYPE_INVALID,
                           TransposeOperation::NONE,
                           TransposeOperation::TRANSPOSE});

    printBufferData(*qk_output, "qk_output: ");

    float scale             = (1.0f / sqrtf(size_per_head * 1.0f));
    auto  softmax_qk_output = softmax({qk_output, std::nullopt, std::nullopt, scale});
    printBufferData(*softmax_qk_output, "softmax_qk_output: ");

    auto qkv_output = gemm({*softmax_qk_output, *v_with_cache});

    transposeDim12(qkv_output, params.output);

    printBufferData(*qkv_output, "decoder qkv_output: ");
    printBufferData(params.output, "params.output: ");
}

/* Inits array as per [batch, seq_len] */
void getPerSeqArray(float *qkv, float **q_array, float **k_array, float **v_array,
                int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_size) {
    for (int i = 0; i < batch_size * seq_len; i++) {
        q_array[i] = qkv + i * (num_heads + 2 * kv_num_heads) * head_size;
        k_array[i] = qkv + i * (num_heads + 2 * kv_num_heads) * head_size + num_heads * head_size;
        v_array[i] = qkv + i * (num_heads + 2 * kv_num_heads) * head_size + (num_heads + kv_num_heads) * head_size;
    }
}

void addBiasArray(float **q_array, float **k_array, float **v_array,
                float *bias,
                int batch, int seq_len, int num_heads, int kv_num_heads, int head_size) {
    const float* q_bias = bias;
    const float* k_bias = bias + num_heads * head_size;
    const float* v_bias = bias + (num_heads + kv_num_heads) * head_size;

    const int N = batch * seq_len;
    parallel_for(N, [&](int tid) {
        for (int i = 0; i < num_heads * head_size; i++) {
            q_array[tid][i] += q_bias[i];
            k_array[tid][i] += k_bias[i];
            v_array[tid][i] += v_bias[i];
        }
    });
}

void contextRopeArray(float **q_array, float **k_array,
                    int batch, int seq_len, int num_heads, int head_size,
                    size_t base, size_t embed_dim) {
    size_t inv_freq_size = (embed_dim + 1) / 2;

    const int N = batch * seq_len;
    parallel_for(N, [&](int tid) {
        int j = tid % seq_len;
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < inv_freq_size; d++) {
                float freq = 1.0f / powf(base, (float)(d * 2) / embed_dim);
                float val  = freq * j;
                float fcr  = cosf(val);
                float fci  = sinf(val);
                auto  v0 = q_array[tid] + h * head_size + d;
                auto  v1 = q_array[tid] + h * head_size + d + inv_freq_size;
                auto  v2 = k_array[tid] + h * head_size + d;
                auto  v3 = k_array[tid] + h * head_size + d + inv_freq_size;
                auto  d0 = *v0;
                auto  d1 = *v1;
                auto  d2 = *v2;
                auto  d3 = *v3;
                *v0  = d0 * fcr - d1 * fci;
                *v1  = d0 * fci + d1 * fcr;
                *v2  = d2 * fcr - d3 * fci;
                *v3  = d2 * fci + d3 * fcr;
            }
        }
    });
}

/* Inits array as per [batch, num_heads] */
void getPerHeadArray(float *qkv, float **q_array, float **k_array, float **v_array,
                int batch_size, int seq_len, int num_heads, int kv_num_heads, int head_size) {
    const int N = batch_size * num_heads;
    parallel_for(N, [&](int tid) {
        int b = tid / num_heads;
        int h = tid % num_heads;
        q_array[tid] = qkv + b * (num_heads + 2 * kv_num_heads) * head_size + h * head_size;
        k_array[tid] = qkv + num_heads * head_size + b * (num_heads + 2 * kv_num_heads) * head_size + h * head_size;
        v_array[tid] = qkv + (num_heads + kv_num_heads) * head_size + b * (num_heads + 2 * kv_num_heads) * head_size + h * head_size;
    });
}

void writeContextKVCacheArray(const AttentionModuleParams& params, float **k_array, float **v_array) {
    const auto batch_size = params.common.context_batch_size;
    const auto seq_len    = params.common.context_max_seq_len;

    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    auto block_tokens  = params.configs.tokens_per_block;

    size_t block_num = (seq_len + block_tokens - 1) / block_tokens;
    auto   elem_sz   = params.input.typeSize();
    size_t copied_len;

    void *k_block_addr;
    void *v_block_addr;

    for (int batch = 0; batch < batch_size; batch++) {
        copied_len = 0;
        for (int i = 0; i < block_num; i++) {
            const size_t   len          = std::min(block_tokens, seq_len - copied_len);
            getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

            parallel_for(len, [&](int tid) {
                memcpy((char*)k_block_addr + tid * elem_sz * kv_head_num * size_per_head,
                       k_array[batch * seq_len + i * block_tokens + tid],
                       elem_sz * 1 * kv_head_num * size_per_head);
                memcpy((char*)v_block_addr + tid * elem_sz * kv_head_num * size_per_head,
                       v_array[batch * seq_len + i * block_tokens + tid],
                       elem_sz * 1 * kv_head_num * size_per_head);
            });

            copied_len += len;
        }
    }
}

#define MAX_CONTEXT_SEQ_SZ  64
void ArmCpuDevice::contextAttentionStride(const AttentionModuleParams& params) {
    auto datatype      = params.input.type();
    auto batch_size    = params.common.context_batch_size;
    auto seq_len       = params.common.context_max_seq_len;
    auto head_num      = params.configs.head_num;
    auto kv_head_num   = params.configs.kv_head_num;
    auto size_per_head = params.configs.size_per_head;
    std::chrono::steady_clock::time_point tStart, tEnd;
    std::chrono::microseconds diff;

    tStart = std::chrono::steady_clock::now();
    // Retrieve q/k/v by stride and not to split. 
    auto qkv = params.input.data();
    printBufferData(params.input, "qkv");

    if (datatype != DataType::TYPE_FP32) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    float *q_array[MAX_CONTEXT_SEQ_SZ], *k_array[MAX_CONTEXT_SEQ_SZ], *v_array[MAX_CONTEXT_SEQ_SZ];
    getPerSeqArray((float *)qkv, q_array, k_array, v_array,
                    batch_size, seq_len, head_num, kv_head_num, size_per_head);
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 0);

    tStart = std::chrono::steady_clock::now();
    if (params.weights.qkv_weight->bias) {
        addBiasArray(q_array, k_array, v_array,
                    (float *)params.weights.qkv_weight->bias->data(),
                    batch_size, seq_len, head_num, kv_head_num, size_per_head);
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 1);

    tStart = std::chrono::steady_clock::now();
    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {
            contextRopeArray(q_array, k_array, batch_size, seq_len, head_num, size_per_head,
                            params.configs.rope_config.base,
                            params.configs.rope_config.dim);
        } else {
            throw std::runtime_error("SelfAttention RoPE type is not supported");
        }
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 2);

    tStart = std::chrono::steady_clock::now();

    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 3);

    printBufferData(params.input, "qkv updated");

    tStart = std::chrono::steady_clock::now();
    if (params.common.kv_cache) {
        writeContextKVCacheArray(params, k_array, v_array);
    }
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 8);

    tStart = std::chrono::steady_clock::now();
    /* Re-init arrary as per [batch, num_heads]. */
    getPerHeadArray((float *)qkv, q_array, k_array, v_array,
                        batch_size, seq_len, head_num, kv_head_num, size_per_head);

    auto qk_output = allocateBuffer({TYPE_FP32, {batch_size, head_num, seq_len, seq_len}, AllocationType::HOST}, {"qk_output"});
    const int stride = (head_num + 2 * kv_head_num) * size_per_head;
    const int N = batch_size * head_num;
    parallel_for(N, [&](int tid) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len, seq_len, size_per_head, 1.0,
            q_array[tid], stride,
            k_array[tid], stride, 0.0,
            (float*)qk_output->dataWithOffset(tid * seq_len * seq_len), seq_len);
    });

    printBufferData(*qk_output, "qk_output: ");
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 4);

    tStart = std::chrono::steady_clock::now();
    float scale = (1.0f / sqrtf(size_per_head * 1.0f));

    RUNTIME_ASSERT_OP_ARG(params.common.attention_mask,
                          "attention_mask must be provided for default context attention implementation");
    printBufferData(*params.common.attention_mask, "attention_mask: ");
    auto softmax_qk_output = softmax({std::move(qk_output), *params.common.attention_mask, std::nullopt, scale});
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 5);

    printBufferData(*softmax_qk_output, "softmax_qk_output: ");

    tStart = std::chrono::steady_clock::now();
    const int NN = batch_size * head_num;
    parallel_for(NN, [&](int tid) {
        const int b = tid / head_num;
        const int h = tid % head_num;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, seq_len, size_per_head, seq_len, 1.0,
            (const float*)softmax_qk_output->dataWithOffset(tid * seq_len * seq_len), seq_len,
            v_array[tid], stride, 0.0,
            (float*)params.output.dataWithOffset(b * seq_len * head_num * size_per_head + h * size_per_head), head_num * size_per_head);
    });
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 6);

    tStart = std::chrono::steady_clock::now();
    tEnd = std::chrono::steady_clock::now();
    diff = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart);
    logTime(diff, 7);

    // /* Print profile data at the end of operator unit test. */
    // if (a_cnt_[0] == 1000)
    //     printStat();
    printBufferData(params.output, "params.output: ");
}

void assemCacheArray(const AttentionModuleParams& params, BufferPtr k_out, BufferPtr v_out, size_t tokens_per_block) {
    auto   elem_sz          = k_out->typeSize();
    auto   batch_size       = k_out->shape()[0];
    auto   head_num         = k_out->shape()[2];
    auto   kv_seq_len       = k_out->shape()[1];
    auto   head_dim         = k_out->shape()[3];
    size_t blocks_per_batch = (kv_seq_len + tokens_per_block - 1) / tokens_per_block;
    size_t copied_len;

    void *k_block_addr;
    void *v_block_addr;
    for (int batch = 0; batch < batch_size; batch++) {
        copied_len = 0;
        for (int i = 0; i < blocks_per_batch; i++) {
            size_t len = std::min(tokens_per_block, kv_seq_len - copied_len);
            getCacheAddrFromIndex(params.common.kv_cache.value(), batch, i, &k_block_addr, &v_block_addr);

            memcpy(k_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   k_block_addr,
                   elem_sz * len * head_num * head_dim);
            memcpy(v_out->dataWithOffset((batch * kv_seq_len + i * tokens_per_block) * head_num * head_dim),
                   v_block_addr,
                   elem_sz * len * head_num * head_dim);
            copied_len += len;
        }
    }
}

void ArmCpuDevice::decoderSelfAttentionStride(const AttentionModuleParams& params) {
    auto   datatype      = params.input.type();
    auto   batch_size    = params.common.decoder_batch_size;
    size_t kv_seq_len    = *static_cast<int*>(params.common.input_lengths.dataWithOffset(0));
    size_t seq_len       = *static_cast<int*>(params.common.sequence_lengths.dataWithOffset(0)) + 1 - kv_seq_len;
    auto   head_num      = params.configs.head_num;
    auto   kv_head_num   = params.configs.kv_head_num;
    auto   size_per_head = params.configs.size_per_head;
    auto   block_tokens  = params.configs.tokens_per_block;

    if (!params.common.kv_cache.has_value()) {
        throw std::runtime_error("kv cache block pointers can not be null");
    }

    // qkv to q_output, k_output, v_output
    auto qkv = params.input.data();

    printBufferData(params.input, "decoder qkv");
    arm_compute::DataType acl_data_type = getAclDataType(datatype);

    arm_compute::NESplit    split;
    arm_compute::Tensor     src;
    arm_compute::TensorInfo src_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num + 2 * kv_head_num, seq_len, batch_size), 1, acl_data_type);
    ;
    src.allocator()->init(src_info);
    src.allocator()->import_memory(qkv);
    std::vector<arm_compute::Tensor>   dsts{};
    std::vector<arm_compute::ITensor*> dsts_ptr;

    arm_compute::TensorInfo q_info, kv_info;
    arm_compute::Tensor     q, k, v;
    q_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, head_num, seq_len, batch_size), 1, acl_data_type);
    kv_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, seq_len, batch_size), 1, acl_data_type);

    auto q_input =
        allocateBuffer({params.input.type(), {batch_size, seq_len, head_num, size_per_head}, AllocationType::HOST}, {});

    auto k_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});

    auto v_input = allocateBuffer(
        {params.input.type(), {batch_size, seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    q.allocator()->init(q_info);
    k.allocator()->init(kv_info);
    v.allocator()->init(kv_info);
    q.allocator()->import_memory(q_input->data());
    dsts.push_back(std::move(q));

    k.allocator()->import_memory(k_input->data());
    dsts.push_back(std::move(k));

    v.allocator()->import_memory(v_input->data());
    dsts.push_back(std::move(v));

    for (auto& dst : dsts) {
        dsts_ptr.emplace_back(&dst);
    }
    split.configure(&src, dsts_ptr, 1);
    split.run();

    printBufferData(*q_input, "q_input");
    printBufferData(*k_input, "k_input");
    printBufferData(*v_input, "v_input");

    if (params.weights.qkv_weight->bias) {
        if (datatype == DataType::TYPE_FP32) {
            addQKVBias<float>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<float>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<float>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else if (datatype == DataType::TYPE_FP16) {
            addQKVBias<__fp16>(*q_input, params.weights.qkv_weight->bias->dataWithOffset(0));
            addQKVBias<__fp16>(*k_input, params.weights.qkv_weight->bias->dataWithOffset(head_num * size_per_head));
            addQKVBias<__fp16>(
                *v_input, params.weights.qkv_weight->bias->dataWithOffset((head_num + kv_head_num) * size_per_head));
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }

    printBufferData(*q_input, "biased q_input");
    printBufferData(*k_input, "biased k_input");
    printBufferData(*v_input, "biased v_input");

    if (params.configs.rope_config.style != RopeStyle::No) {
        if (params.configs.rope_config.style == RopeStyle::Base) {

            if (datatype == DataType::TYPE_FP32) {
                attention_rope<float>(*q_input,
                                      seq_len + kv_seq_len - 1,
                                      params.configs.rope_config.base,
                                      params.configs.rope_config.dim);
                attention_rope<float>(*k_input,
                                      seq_len + kv_seq_len - 1,
                                      params.configs.rope_config.base,
                                      params.configs.rope_config.dim);
            } else if (datatype == DataType::TYPE_FP16) {
                attention_rope<__fp16>(*q_input,
                                       seq_len + kv_seq_len - 1,
                                       params.configs.rope_config.base,
                                       params.configs.rope_config.dim);
                attention_rope<__fp16>(*k_input,
                                       seq_len + kv_seq_len - 1,
                                       params.configs.rope_config.base,
                                       params.configs.rope_config.dim);
            } else {
                throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
            }
        } else {
            throw std::runtime_error("SelfAttention RoPE type is not supported");
        }
    }

    updateKVCache(params, k_input, v_input);

    auto q_output =
        allocateBuffer({params.input.type(), {batch_size, head_num, seq_len, size_per_head}, AllocationType::HOST}, {});

    transposeDim12(std::move(q_input), *q_output);

    /* Retrieve k/v cache data and attach it to k/v tensor. */
    auto k_cache =
        allocateBuffer({datatype, {batch_size, kv_seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    auto v_cache =
        allocateBuffer({datatype, {batch_size, kv_seq_len, kv_head_num, size_per_head}, AllocationType::HOST}, {});
    assemCacheArray(params, k_cache, v_cache, block_tokens);

    printBufferData(*k_cache, "k_cache");
    printBufferData(*v_cache, "v_cache");
    auto k_with_cache = allocateBuffer(
        {params.input.type(), {batch_size, seq_len + kv_seq_len, kv_head_num, size_per_head}, AllocationType::HOST},
        {});
    auto v_with_cache = allocateBuffer(
        {params.input.type(), {batch_size, seq_len + kv_seq_len, kv_head_num, size_per_head}, AllocationType::HOST},
        {});
    arm_compute::NEConcatenateLayer          cat;
    std::vector<arm_compute::Tensor>         kv_inputs{};
    std::vector<const arm_compute::ITensor*> kv_inputs_vector;
    arm_compute::Tensor                      kv_cache, kv_cat;
    arm_compute::TensorInfo                  kv_cache_info, kv_cat_info;
    kv_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, seq_len, batch_size), 1, acl_data_type);
    kv_cache_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, kv_seq_len, batch_size), 1, acl_data_type);
    kv_cat_info = arm_compute::TensorInfo(
        arm_compute::TensorShape(size_per_head, kv_head_num, kv_seq_len + seq_len, batch_size), 1, acl_data_type);
    kv_cache.allocator()->init(kv_cache_info);
    kv_cache.allocator()->import_memory(k_cache->data());
    kv_inputs.push_back(std::move(kv_cache));
    k.allocator()->init(kv_info);
    k.allocator()->import_memory(k_input->data());
    kv_inputs.push_back(std::move(k));
    for (auto& input : kv_inputs) {
        kv_inputs_vector.emplace_back(&input);
    }
    kv_cat.allocator()->init(kv_cat_info);
    kv_cat.allocator()->import_memory(k_with_cache->data());
    cat.configure(kv_inputs_vector, &kv_cat, 2);
    cat.run();

    kv_inputs.clear();
    kv_inputs_vector.clear();

    kv_cache.allocator()->init(kv_cache_info);
    kv_cache.allocator()->import_memory(v_cache->data());
    kv_inputs.push_back(std::move(kv_cache));
    v.allocator()->init(kv_info);
    v.allocator()->import_memory(v_input->data());
    kv_inputs.push_back(std::move(v));
    for (auto& input : kv_inputs) {
        kv_inputs_vector.emplace_back(&input);
    }
    kv_cat.allocator()->init(kv_cat_info);
    kv_cat.allocator()->import_memory(v_with_cache->data());
    cat.configure(kv_inputs_vector, &kv_cat, 2);
    cat.run();
    printBufferData(*k_with_cache, "k_with_cache");
    printBufferData(*v_with_cache, "v_with_cache");

    auto k_with_cache_trans12 = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len + kv_seq_len, size_per_head}, AllocationType::HOST},
        {});
    auto v_with_cache_trans12 = allocateBuffer(
        {params.input.type(), {batch_size, kv_head_num, seq_len + kv_seq_len, size_per_head}, AllocationType::HOST},
        {});
    transposeDim12(std::move(k_with_cache), *k_with_cache_trans12);
    transposeDim12(std::move(v_with_cache), *v_with_cache_trans12);
    auto qk_output = gemm({*q_output,
                           *k_with_cache_trans12,
                           std::nullopt,
                           nullptr,
                           DataType::TYPE_INVALID,
                           TransposeOperation::NONE,
                           TransposeOperation::TRANSPOSE});

    printBufferData(*qk_output, "qk_output: ");

    float scale             = (1.0f / sqrtf(size_per_head * 1.0f));
    auto  softmax_qk_output = softmax({qk_output, std::nullopt, std::nullopt, scale});
    printBufferData(*softmax_qk_output, "softmax_qk_output: ");

    auto qkv_output = gemm({*softmax_qk_output, *v_with_cache_trans12});

    transposeDim12(qkv_output, params.output);

    printBufferData(*qkv_output, "decoder qkv_output: ");
    printBufferData(params.output, "params.output: ");
}

AttentionModuleOutput ArmCpuDevice::contextAttention(const AttentionModuleParams& params) {
    if (params.input.type() == DataType::TYPE_FP32) {
        contextAttentionStride(params);
    } else if (params.input.type() == DataType::TYPE_FP16) {
        contextAttentionFallback(params);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

AttentionModuleOutput ArmCpuDevice::decoderSelfAttention(const AttentionModuleParams& params) {
    if (params.input.type() == DataType::TYPE_FP32) {
        decoderSelfAttentionStride(params);
    } else if (params.input.type() == DataType::TYPE_FP16) {
        decoderSelfAttentionFallback(params);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}
}  // namespace fastertransformer

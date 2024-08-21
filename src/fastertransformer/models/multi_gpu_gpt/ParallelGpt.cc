#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include <algorithm>
#include <stdio.h>

namespace fastertransformer {

template<typename T>
void ParallelGpt<T>::initialize()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // 默认stdout输出到文件的逻辑是全缓冲，导致ft_log和autil_log日志刷不出来，手动设置为行缓冲
    setlinebuf(stdout);
    quant_algo_                 = tensorrt_llm::common::QuantAlgo(params_.quant_algo_);
    parallel_attention_wrapper_ = new ParallelAttentionWrapper<T>(params_,
                                                                  tensor_para_,
                                                                  stream_,
                                                                  cublas_wrapper_,
                                                                  quant_algo_,
                                                                  allocator_,
                                                                  is_free_buffer_after_forward_,
                                                                  is_qk_buf_float_,
                                                                  sparse_);

    // max_seq_len + max_generate_batch_size because max_seq_len >> max_generate_batch_size
    ffn_layer_ = new TensorParallelFfnLayer<T>(params_.max_context_batch_size_,
                                               params_.max_seq_len_ + params_.max_generate_batch_size_,
                                               params_.hidden_size_,
                                               params_.expert_num_,  // expert_num
                                               params_.moe_k_,
                                               params_.moe_normalize_expert_scale_,
                                               params_.moe_style_,
                                               params_.inter_size_,
                                               params_.inter_padding_size_,
                                               params_.moe_inter_padding_size_,
                                               params_.layer_inter_size_,
                                               params_.layer_inter_padding_size_,
                                               tensor_para_,
                                               stream_,
                                               cublas_wrapper_,
                                               quant_algo_,
                                               allocator_,
                                               true,
                                               is_free_buffer_after_forward_,
                                               sparse_,
                                               params_.is_sparse_head_,
                                               params_.activation_type_,
                                               params_.has_moe_norm_,
                                               params_.layernorm_eps_,
                                               custom_all_reduce_comm_,
                                               enable_custom_all_reduce_);

    norm_wrapper_.reset(
        new NormWrapper<T>(params_.layernorm_type_, params_.norm_type_, T(sqrt(2 * params_.num_layers_))));
}

template<typename T>
void ParallelGpt<T>::preAllocate()
{
    parallel_attention_wrapper_->preAllocate();
    ffn_layer_->preAllocate();
    allocateBuffer(
        params_.max_generate_batch_size_ + params_.max_context_batch_size_, params_.max_seq_len_, false, true);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer(size_t total_batch_size, size_t h_token_num, bool reuse_buf, bool pre_attn_ln)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t hidden_units   = params_.hidden_size_;
    const int max_blocks_per_batch = params_.max_seq_len_ / params_.seq_size_per_block_ + 1;
    decoder_normed_input_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_normed_input_, sizeof(T) * h_token_num * hidden_units));
    self_attn_output_ =
        reinterpret_cast<T*>(allocator_->reMalloc(self_attn_output_, sizeof(T) * h_token_num * hidden_units));
    if (!reuse_buf) {
        normed_self_attn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(normed_self_attn_output_, sizeof(T) * h_token_num * hidden_units));
    }
    else {
        normed_self_attn_output_ = decoder_normed_input_;
    }
    if (pre_attn_ln) {
        attn_normed_input_ = reinterpret_cast<T*>(
            allocator_->reMalloc(attn_normed_input_, sizeof(T) * h_token_num * hidden_units));
    }
    // only allocate additionl buffers when has adapters
    decoder_layer_output_ = reinterpret_cast<T*>(
        allocator_->reMalloc(decoder_layer_output_, sizeof(T) * h_token_num * hidden_units));
    if (quant_algo_.smoothQuantInt8()) {
        attention_query_dynamic_scale_ = reinterpret_cast<float*>(
            allocator_->reMalloc(attention_query_dynamic_scale_, sizeof(float) * h_token_num));
        ffn_intermediate_dynamic_scale_ = reinterpret_cast<float*>(
            allocator_->reMalloc(ffn_intermediate_dynamic_scale_, sizeof(float) * h_token_num));
    }
    padding_offset_ = reinterpret_cast<int*>(allocator_->reMalloc(padding_offset_, sizeof(int) * (h_token_num)));
    cu_seqlens_ =
        reinterpret_cast<int*>(allocator_->reMalloc(cu_seqlens_, sizeof(int) * (total_batch_size + 1)));
    cu_kv_seqlens_ =
        reinterpret_cast<int*>(allocator_->reMalloc(cu_kv_seqlens_, sizeof(int) * (total_batch_size + 1)));
    context_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(context_lengths_, sizeof(int) * (total_batch_size)));
    sequence_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(sequence_lengths_, sizeof(int) * (total_batch_size)));
    prefix_lengths_ =
        reinterpret_cast<int*>(allocator_->reMalloc(prefix_lengths_, sizeof(int) * (total_batch_size)));
    block_pointers_ =
        reinterpret_cast<uint64_t*>(allocator_->reMalloc(block_pointers_, sizeof(uint64_t) * (2 * params_.num_layers_ * total_batch_size * max_blocks_per_batch)));
    block_offset_ = reinterpret_cast<int*>(allocator_->reMalloc(block_offset_, sizeof(int) * (total_batch_size * max_blocks_per_batch)));
    k_cache_base_addr_ = reinterpret_cast<uint64_t*>(allocator_->reMalloc(k_cache_base_addr_, sizeof(uint64_t) * (4 * params_.num_layers_)));
    v_cache_base_addr_ = k_cache_base_addr_ + params_.num_layers_;
    k_scale_base_addr_ = k_cache_base_addr_ + 2 * params_.num_layers_;
    v_scale_base_addr_ = k_cache_base_addr_ + 3 * params_.num_layers_;
    if (params_.int8_kv_cache_) {
        block_scale_pointers_ =
            reinterpret_cast<uint64_t*>(allocator_->reMalloc(block_scale_pointers_, sizeof(uint64_t) * (2 * params_.num_layers_ * total_batch_size * max_blocks_per_batch)));
    }

    // for moe
    expert_scales_ = reinterpret_cast<float*>(
        allocator_->reMalloc(expert_scales_, sizeof(float) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num)));
    expanded_source_row_to_expanded_dest_row_ = reinterpret_cast<int*>(allocator_->reMalloc(
        expanded_source_row_to_expanded_dest_row_, sizeof(int) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num)));
    expert_for_source_row_                    = reinterpret_cast<int*>(
        allocator_->reMalloc(expert_for_source_row_, sizeof(int) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num)));
    fc2_result_ = reinterpret_cast<T*>(
        allocator_->reMalloc(fc2_result_, sizeof(T) * pad_to_multiple_of_16(params_.moe_k_ * h_token_num * hidden_units)));
    if (params_.moe_style_ == 2) {
        partial_moe_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(partial_moe_output_, sizeof(T) * h_token_num * hidden_units));
        ffn_output_ = reinterpret_cast<T*>(
            allocator_->reMalloc(ffn_output_, sizeof(T) * h_token_num * hidden_units));
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        if (normed_self_attn_output_ != decoder_normed_input_) {
            allocator_->free((void**)(&normed_self_attn_output_));
        }
        if (attn_normed_input_) {
            allocator_->free((void**)(&attn_normed_input_));
        }
        allocator_->free((void**)(&decoder_normed_input_));
        allocator_->free((void**)(&self_attn_output_));
        allocator_->free((void**)(&decoder_layer_output_));
        allocator_->free((void**)(&padding_offset_));
        allocator_->free((void**)(&cu_seqlens_));
        allocator_->free((void**)(&cu_kv_seqlens_));
        allocator_->free((void**)(&context_lengths_));
        allocator_->free((void**)(&sequence_lengths_));
        allocator_->free((void**)(&prefix_lengths_));
        allocator_->free((void**)(&block_pointers_));
        allocator_->free((void**)(&block_scale_pointers_));
        allocator_->free((void**)(&k_cache_base_addr_));
        if (quant_algo_.smoothQuantInt8()) {
            allocator_->free((void**)(&attention_query_dynamic_scale_));
            allocator_->free((void**)(&ffn_intermediate_dynamic_scale_));
        }
        allocator_->free((void**)(&expert_scales_));
        allocator_->free((void**)(&expanded_source_row_to_expanded_dest_row_));
        allocator_->free((void**)(&expert_for_source_row_));
        allocator_->free((void**)(&fc2_result_));
        if(params_.moe_style_ == 2) {
            allocator_->free((void**)(&partial_moe_output_));
            allocator_->free((void**)(&ffn_output_));
        }
        is_allocate_buffer_ = false;
    }
}

template<typename T>
bool ParallelGpt<T>::isValidLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l >= local_num_layer * pipeline_para_.rank_)
           && (l < local_num_layer * (pipeline_para_.rank_ + 1));
}

template<typename T>
bool ParallelGpt<T>::isFirstLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l == local_num_layer * pipeline_para_.rank_);
}

template<typename T>
bool ParallelGpt<T>::isLastLayerParallelId(uint l)
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return l < params_.num_layers_ && (l == local_num_layer * (pipeline_para_.rank_ + 1) - 1);
}

template<typename T>
int ParallelGpt<T>::getFirstLayerParallelId()
{
    uint local_num_layer = (uint)(ceil(params_.num_layers_ * 1.0f / pipeline_para_.world_size_));
    return local_num_layer * pipeline_para_.rank_;
}

template<typename T>
ParallelGpt<T>::ParallelGpt(const GptInitParameter&             gpt_init_parameter,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            bool                                is_qk_buf_float,
                            bool                                sparse,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr, sparse),
    params_(gpt_init_parameter),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    is_qk_buf_float_(is_qk_buf_float)
{
    initialize();
}

template<typename T>
bool ParallelGpt<T>::UseFMHA()
{
    FT_CHECK_WITH_INFO(parallel_attention_wrapper_ != nullptr, "parallel_attention_wrapper_ should not be nullptr");
    return parallel_attention_wrapper_->UseFMHA();
}

template<typename T>
ParallelGpt<T>::~ParallelGpt()
{
    delete parallel_attention_wrapper_;
    delete ffn_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGpt<T>::convert_to_block_pointers(TensorMap* output_tensors,
                                               const TensorMap* input_tensors,
                                               int total_batch_size)
{
    block_base_addr_vector_.clear();
    Tensor block_index_map      = input_tensors->at("block_index_map");
    uint   max_blocks_per_batch = (uint)(input_tensors->at("block_index_map").shape()[1]);
    assert(max_blocks_per_batch <= params_.max_seq_len_ / params_.seq_size_per_block_);
    block_base_addr_vector_.resize(params_.num_layers_ * 4, 0);
    Tensor k_cache         = output_tensors->at("key_cache");
    Tensor v_cache         = output_tensors->at("value_cache");
    size_t kv_cache_offset = 1;
    for (auto t = k_cache.shape().begin() + 1; t != k_cache.shape().end(); ++t) {
        kv_cache_offset *= *t;
    };
    for (uint l = 0; l < params_.num_layers_; l++) {
        const size_t cache_offset  = (l - getFirstLayerParallelId()) * kv_cache_offset;
        block_base_addr_vector_[l] = (uint64_t)k_cache.getPtrWithOffset(cache_offset);
        block_base_addr_vector_[l + params_.num_layers_] = (uint64_t)v_cache.getPtrWithOffset(cache_offset);
    }
    if (params_.int8_kv_cache_) {
        Tensor k_cache_scale = output_tensors->at("key_cache_scale");
        Tensor v_cache_scale = output_tensors->at("value_cache_scale");
        for (uint l = 0; l < params_.num_layers_; l++) {
            const size_t cache_offset        = (l - getFirstLayerParallelId()) * kv_cache_offset;
            const size_t scale_cache_offset  = cache_offset / params_.size_per_head_;
            block_base_addr_vector_[2 * params_.num_layers_ + l] = (uint64_t)k_cache_scale.getPtrWithOffset<float>(scale_cache_offset);
            block_base_addr_vector_[2 * params_.num_layers_ + l + params_.num_layers_] = (uint64_t)v_cache_scale.getPtrWithOffset<float>(scale_cache_offset);
        }
    }
    cudaMemcpyAsync(block_offset_, block_index_map.data(), block_index_map.sizeBytes(), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(k_cache_base_addr_, block_base_addr_vector_.data(), sizeof(uint64_t) * block_base_addr_vector_.size(), cudaMemcpyHostToDevice, stream_);
    invokeConvertOffsetToAddr(
            block_pointers_,
            k_cache_base_addr_,
            v_cache_base_addr_,
            block_offset_,
            params_.num_layers_,
            total_batch_size,
            max_blocks_per_batch,
            k_cache.sizeBytes() / k_cache.shape()[0] / k_cache.shape()[1],
            stream_);
    if (params_.int8_kv_cache_) {
        Tensor k_scale = output_tensors->at("key_cache_scale");
        invokeConvertOffsetToAddr(
                block_scale_pointers_,
                k_scale_base_addr_,
                v_scale_base_addr_,
                block_offset_,
                params_.num_layers_,
                total_batch_size,
                max_blocks_per_batch,
                k_scale.sizeBytes() / k_scale.shape()[0] / k_scale.shape()[1],
                stream_);
    }
}

template<typename T>
void ParallelGpt<T>::forward(TensorMap*                                            output_tensors,
                             const TensorMap*                                      input_tensors,
                             const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight)
{
    // input tensors:
    //      decoder_input [batch_size + context_batch_size, hidden_dimension],
    //      attention_mask [context_batch_size, 1, seq_len, seq_len]
    //      input_lengths [batch_size + context_batch_size]
    //      sequence_lengths [batch_size]
    //      block_index_map [batch_size + context_batch_size, max_block_size]
    //      linear_bias_slopes [head_num], optional

    // output tensors:
    //      decoder_output [batch_size + context_batch_size, hidden_dimension]
    //      key_cache [num_layer, batch_size, head_num, params_.size_per_head_ // x, memory_len, x]
    //      value_cache [num_layer, batch_size, head_num, memory_len, params_.size_per_head_]

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->isExist("decoder_input"));
    FT_CHECK(input_tensors->isExist("input_lengths"));
    FT_CHECK(output_tensors->isExist("decoder_output"));
    FT_CHECK(input_tensors->isExist("lora_ids"));
    FT_CHECK(input_tensors->isExist("lora_input_lengths"));


    Tensor decoder_input_tensor = input_tensors->at("decoder_input");
    T* decoder_output_ptr = output_tensors->at("decoder_output").getPtr<T>();
    size_t hidden_units         = params_.hidden_size_;
    FT_CHECK(decoder_input_tensor.shape()[1] == hidden_units);
    const size_t total_batch_size = input_tensors->at("input_lengths").shape()[0];
    size_t       batch_size       = 0;
    if (input_tensors->isExist("sequence_lengths")) {
        batch_size = input_tensors->at("sequence_lengths").shape()[0];
    }
    const size_t   h_token_num = decoder_input_tensor.shape()[0];
    const DataType data_type   = getTensorType<T>();
    const bool     use_expert_attention = input_tensors->isExist("token_type_ids");

    PUSH_RANGE(stream_, "buffer allocation");
    bool reuse_buf   = !params_.use_norm_input_residual_;
    bool pre_attn_ln = gpt_decoder_layer_weight->at(0)->pre_attn_layernorm_weights.gamma;
    allocateBuffer(total_batch_size, h_token_num, reuse_buf, pre_attn_ln);
    POP_RANGE;

    const size_t context_batch_size = total_batch_size - batch_size;

    int*         input_lengths      = input_tensors->getPtr<int>("input_lengths");
    cudaMemcpyAsync(context_lengths_, input_lengths, sizeof(int) * total_batch_size, cudaMemcpyHostToDevice, stream_);
    size_t max_input_length = 0;
    size_t max_context_seq_length = 0;
    size_t step             = 0;
    if (context_batch_size) {
        max_context_seq_length = *std::max_element(input_lengths + batch_size, input_lengths + total_batch_size);
        if (input_tensors->isExist("attention_mask")) {
            FT_CHECK(input_tensors->at("attention_mask").shape()[0] == context_batch_size);
        }
    }
    if (batch_size) {
        int* sequence_lengths = input_tensors->getPtr<int>("sequence_lengths");
        cudaMemcpyAsync(sequence_lengths_, sequence_lengths, sizeof(int) * batch_size, cudaMemcpyHostToDevice, stream_);
        max_input_length = *std::max_element(input_lengths, input_lengths + batch_size);
        step             = *std::max_element(sequence_lengths, sequence_lengths + batch_size);
        step             = step + 1;
    }

    int max_context_prefix_length = 0;
    if (input_tensors->isExist("d_prefix_prompt_lengths")) {
        int *d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths");
        if (context_batch_size > 0) {
            max_context_prefix_length = *std::max_element(d_prefix_prompt_lengths + batch_size, d_prefix_prompt_lengths + total_batch_size);
        }
        cudaMemcpyAsync(prefix_lengths_, d_prefix_prompt_lengths, sizeof(int) * total_batch_size,
                        cudaMemcpyHostToDevice, stream_);
    }

    uint   max_blocks_per_batch = 0;
    size_t block_stride = 0;
    if (params_.use_kvcache_) {
        if (output_tensors->isExist("block_pointers")) {
            Tensor block_pointers = output_tensors->at("block_pointers");
            FT_CHECK(int(block_pointers.shape()[0]) == params_.num_layers_);
            FT_CHECK(block_pointers.shape()[1] == total_batch_size);
            FT_CHECK(block_pointers.shape()[2] == 2);
            max_blocks_per_batch = block_pointers.shape()[3];
            block_stride = total_batch_size * 2 * max_blocks_per_batch;
            size_t data_nums = params_.num_layers_ * total_batch_size * max_blocks_per_batch * 2;
            size_t data_size = sizeof(uint64_t) * data_nums;
            cudaMemcpyAsync(block_pointers_, block_pointers.data(), data_size, cudaMemcpyHostToDevice, stream_);
            if (params_.int8_kv_cache_) {
                Tensor block_scale_pointers = output_tensors->at("block_scale_pointers");
                cudaMemcpyAsync(block_scale_pointers_, block_scale_pointers.data(), data_size, cudaMemcpyHostToDevice, stream_);
            }
        } else {
            convert_to_block_pointers(output_tensors, input_tensors, total_batch_size);
            max_blocks_per_batch = (uint)(input_tensors->at("block_index_map").shape()[1]);
            block_stride = total_batch_size * 2 * max_blocks_per_batch;
        }
    }

    const auto activation_in_type  = quant_algo_.smoothQuantInt8() ? TYPE_INT8 : data_type;
    const auto activation_out_type = data_type;

    size_t context_h_token_num = h_token_num - batch_size;
    if (context_batch_size) {
        PUSH_RANGE(stream_, "remove padding");
        invokeGetPaddingOffsetAndCuSeqLens(
            padding_offset_,
            cu_seqlens_,
            context_lengths_ + batch_size,
            context_batch_size,
            max_context_seq_length,
            stream_);
        FT_CHECK_WITH_INFO(context_h_token_num>0, "input should not be empty");
        POP_RANGE;
    }
    PUSH_RANGE(stream_, "context_generation");
    for (uint l = 0; l < params_.num_layers_; l++) {
        PUSH_RANGE(stream_, fmtstr("layer_%u", l));
        bool use_moe = std::find(params_.moe_layer_index_.begin(), params_.moe_layer_index_.end(), l) != params_.moe_layer_index_.end();
        if (isValidLayerParallelId(l) == false) {
            POP_RANGE;  // escape for NVTX Range: layer_%u
            continue;
        }
        ParallelGptDecoderLayerWeight<T>* layer_weight = gpt_decoder_layer_weight->at(l);
        T* decoder_input  = (l == 0) ? decoder_input_tensor.getPtr<T>() : decoder_output_ptr;
        T* decoder_output = decoder_output_ptr;
        sync_check_cuda_error();

        print_bsd(l, "decoder input", decoder_input, 1, h_token_num, hidden_units);
        if (isFirstLayerParallelId(l) && pipeline_para_.rank_ != 0) {
            PUSH_RANGE(stream_, "input communication");
            const int data_size = h_token_num * hidden_units / tensor_para_.world_size_;
            ftNcclRecv(decoder_input + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ - 1,
                       pipeline_para_,
                       stream_);
            if (tensor_para_.world_size_ > 1) {
                PUSH_RANGE(stream_, "all gather");
                ftNcclAllGather(decoder_input, decoder_input, data_size, tensor_para_.rank_, tensor_para_, stream_);
                POP_RANGE;
            }
            POP_RANGE;
        }
        sync_check_cuda_error();

        PUSH_RANGE(stream_, "pre-mha layernorm");
        if (layer_weight->pre_layernorm_weights.gamma) {
            norm_wrapper_->initDecoderLayerNorm(decoder_normed_input_,
                                                decoder_input,
                                                layer_weight->pre_layernorm_weights.gamma,
                                                layer_weight->pre_layernorm_weights.beta,
                                                params_.layernorm_eps_,
                                                h_token_num,
                                                hidden_units,
                                                nullptr,
                                                attention_query_dynamic_scale_,
                                                reinterpret_cast<int8_t*>(decoder_normed_input_),
                                                stream_);
            if (quant_algo_.smoothQuantInt8()) {
                print_bsd(l, "pre ln", reinterpret_cast<int8_t*>(decoder_normed_input_), 1, h_token_num, hidden_units);
            } else {
                print_bsd(l, "pre ln", decoder_normed_input_, 1, h_token_num, hidden_units);
            }
        }
        sync_check_cuda_error();

        if (pre_attn_ln) {
            norm_wrapper_->preAttentionLayerNorm(attn_normed_input_,
                                                 decoder_input,
                                                 layer_weight->pre_attn_layernorm_weights.gamma,
                                                 layer_weight->pre_attn_layernorm_weights.beta,
                                                 params_.layernorm_eps_,
                                                 h_token_num,
                                                 hidden_units,
                                                 nullptr,
                                                 attention_query_dynamic_scale_,
                                                 reinterpret_cast<int8_t*>(attn_normed_input_),
                                                 stream_);
            print_bsd(l, "pre attn ln", attn_normed_input_, 1, h_token_num, hidden_units);
        }

        sync_check_cuda_error();
        POP_RANGE;

        // TODO(xinfei.sxf) 为啥有两个pre ln
        const T* input_query  = nullptr;
        if (pre_attn_ln) {
            input_query = attn_normed_input_;
        }
        else if (params_.layernorm_type_ == LayerNormType::pre_layernorm) {
            input_query = decoder_normed_input_;
        }
        else {
            input_query = decoder_input;
        }

        print_bsd(l, "input query", input_query, 1, h_token_num, hidden_units);

        TensorMap attention_input_tensors{
            {"input_query", Tensor{MEMORY_GPU, activation_in_type, {h_token_num, hidden_units}, input_query}},
            {"block_pointers",
             Tensor{MEMORY_GPU, TYPE_INT64, {total_batch_size, 1, 2, max_blocks_per_batch}, block_pointers_ + l * block_stride}},
            {"block_offset",
             Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size, max_blocks_per_batch}, block_offset_}},
            {"k_cache_base_addr",
             Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &block_base_addr_vector_[l]}},
            {"v_cache_base_addr",
             Tensor{MEMORY_CPU, TYPE_UINT64, {1}, &block_base_addr_vector_[l + params_.num_layers_]}},
            {"block_scale_pointers",
             Tensor{MEMORY_GPU, TYPE_INT64, {total_batch_size, 1, 2, max_blocks_per_batch}, block_scale_pointers_ + l * block_stride}},
            {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
            {"generate_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &batch_size}},
            {"context_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &context_batch_size}},
            {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &max_input_length}},
            {"max_context_seq_length", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &max_context_seq_length}},
            {"step", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &step}},
            {"sequence_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, sequence_lengths_}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, context_lengths_}},
            {"lora_ids", input_tensors->at("lora_ids")},
            {"lora_input_lengths", input_tensors->at("lora_input_lengths")},
            {"use_expert_attention", Tensor{MEMORY_CPU, TYPE_BOOL, {(size_t)1}, &use_expert_attention}}};

        if (use_expert_attention) {
            attention_input_tensors.insert("token_type_ids", input_tensors->at("token_type_ids"));
        }

        if (quant_algo_.smoothQuantInt8()) {
            FT_CHECK_WITH_INFO(attention_query_dynamic_scale_!=nullptr, "attention_query_dynamic_scale_ should not be nullptr");
            attention_input_tensors.insert(
                "attn_dynamic_scale", Tensor{MEMORY_GPU, TYPE_FP32, {h_token_num, 1}, attention_query_dynamic_scale_});
        }

        if (context_batch_size) {
            if (input_tensors->isExist("attention_mask")) {
                const T* attention_ptr    = input_tensors->at("attention_mask").getPtr<T>();
                auto     attention_tensor = input_tensors->at("attention_mask");
                attention_input_tensors.insert(
                    "attention_mask",
                    Tensor{MEMORY_GPU,
                    data_type,
                    {context_batch_size, 1, attention_tensor.shape()[1], attention_tensor.shape()[2]},attention_ptr}
                );
            }
            attention_input_tensors.insert("padding_offset",
                                           Tensor{MEMORY_GPU, TYPE_INT32, {context_h_token_num}, padding_offset_});
            attention_input_tensors.insert(
                "cu_seqlens", Tensor{MEMORY_GPU, TYPE_INT32, {size_t(context_batch_size + 1)}, cu_seqlens_});
        }
        if (input_tensors->isExist("linear_bias_slopes")) {
            attention_input_tensors.insert("linear_bias_slopes", input_tensors->at("linear_bias_slopes"));
        }
        if (input_tensors->isExist("position_ids")) {
            attention_input_tensors.insert("position_ids", input_tensors->at("position_ids"));
        }
        if (input_tensors->isExist("d_prefix_prompt_lengths")) {
            attention_input_tensors.insert(
                    "max_prefix_prompt_length", input_tensors->at("max_prefix_prompt_length")
            );
            attention_input_tensors.insert("d_prefix_prompt_lengths",
                                           Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, prefix_lengths_});
            invokeGetCuSeqLens(cu_kv_seqlens_, context_lengths_ + batch_size, prefix_lengths_ + batch_size, context_batch_size, stream_);
            attention_input_tensors.insert("max_context_prefix_length",  Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_context_prefix_length});
            attention_input_tensors.insert("cu_kv_seqlens",  Tensor{MEMORY_GPU, TYPE_INT32, {context_batch_size + 1}, cu_kv_seqlens_});
            attention_input_tensors.insert("count_prefix_length", input_tensors->at("count_prefix_length"));
        } else {
            attention_input_tensors.insert("cu_kv_seqlens",  Tensor{MEMORY_GPU, TYPE_INT32, {context_batch_size + 1}, cu_seqlens_});
        }
        TensorMap attention_output_tensors{
            {"hidden_features",
             Tensor(MEMORY_GPU, activation_out_type, {h_token_num, hidden_units}, self_attn_output_)}};

        if (params_.is_sparse_head_ && params_.layer_head_num_[l] == 0) {
            check_cuda_error(cudaMemcpyAsync(self_attn_output_,
                                             input_query,
                                             sizeof(T) * h_token_num * hidden_units,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        else {
            parallel_attention_wrapper_->forward(
                &attention_output_tensors, &attention_input_tensors, &layer_weight->self_attention_weights);
        }

        print_bsd(l, "attn out", self_attn_output_, 1, h_token_num, hidden_units);

        // the adapter after attention (only pre layernorm currently)
        PUSH_RANGE(stream_, "post_mha_ln");
        T* input_residual = nullptr;
        if (!layer_weight->self_attn_layernorm_weights.gamma) {
            // falcon7b
            // output = attn(norm1(in)) + mlp(norm1(in)) + in
            // falcon40b
            // output = attn(norm2(in)) + mlp(norm1(in)) + in
            input_residual = decoder_input;
            std::swap(normed_self_attn_output_, decoder_normed_input_);
        }
        else {
            norm_wrapper_->attentionAddBiasResidualLayerNorm(
                self_attn_output_,
                normed_self_attn_output_,
                self_attn_output_,
                params_.use_norm_input_residual_ ? decoder_normed_input_ : decoder_input,
                layer_weight->self_attn_layernorm_weights.gamma,
                layer_weight->self_attn_layernorm_weights.beta,
                layer_weight->self_attention_weights.attention_output_weight.bias,
                params_.layernorm_eps_,
                h_token_num,
                hidden_units,
                nullptr,
                ffn_intermediate_dynamic_scale_,
                reinterpret_cast<int8_t*>(normed_self_attn_output_),
                stream_);
        }
        sync_check_cuda_error();
        POP_RANGE;

        T* ffn_output_ptr = nullptr;
        if (params_.moe_style_ == 2) {
            ffn_output_ptr = ffn_output_;
        }
        else {
            ffn_output_ptr =  params_.layernorm_type_ == LayerNormType::pre_layernorm ? decoder_normed_input_ : decoder_output;
        }

        int ffn_batch_size_lora = batch_size + context_batch_size;
        const int* lora_input_lengths = input_tensors->getPtr<int>("lora_input_lengths", nullptr);;

        if (quant_algo_.smoothQuantInt8()) {
            print_bsd(l,
                      "before ffn",
                      params_.layernorm_type_ == LayerNormType::pre_layernorm ?
                          reinterpret_cast<int8_t*>(normed_self_attn_output_) :
                          reinterpret_cast<int8_t*>(self_attn_output_),
                      1,
                      h_token_num,
                      hidden_units);
        } else {
            print_bsd(l,
                      "before ffn",
                      params_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ :
                                                                                self_attn_output_,
                      1,
                      h_token_num,
                      hidden_units);
        }

        T* ffn_input_ptr = params_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ : self_attn_output_;

        std::unique_ptr<ExpertAttentionUtil<T>> expert_attention_util = nullptr;
        if (use_expert_attention) {
            expert_attention_util = std::make_unique<ExpertAttentionUtil<T>>(&stream_, allocator_, input_tensors->at("token_type_ids").getPtr<int32_t>(), h_token_num, ffn_input_ptr, ffn_output_ptr);
        }
        Ffnforward(expert_attention_util, ffn_input_ptr, ffn_output_ptr, h_token_num, activation_in_type,
            hidden_units, input_tensors, l, total_batch_size, lora_input_lengths, activation_out_type, ffn_batch_size_lora, layer_weight, use_moe, false);

        if (use_expert_attention && expert_attention_util->vision_token_length() > 0) {
            Ffnforward(expert_attention_util, ffn_input_ptr, ffn_output_ptr, h_token_num, activation_in_type, hidden_units, input_tensors,
            l, total_batch_size, lora_input_lengths, activation_out_type, ffn_batch_size_lora, layer_weight, use_moe, true);
            expert_attention_util->reorganize();
        }

        // the adapter after ffn (only pre layernorm currently)
        PUSH_RANGE(stream_, "post ffn");

        // NOTE: here input and residual1 args are reversed,
        // This is for ChatGLM, which uses alpha norm and the alpha is multiplied at residual:
        // `output = mlp_input * alpha + mlp_output`
        // see https://huggingface.co/THUDM/chatglm-6b/blob/8b7d33596d18c5e83e2da052d05ca4db02e60620/modeling_chatglm.py#L651
        norm_wrapper_->ffnAddBiasResidualLayerNorm(decoder_output,
                                                   params_.use_norm_attn_out_residual_ ? normed_self_attn_output_ :
                                                       self_attn_output_,
                                                   ffn_output_ptr,
                                                   params_.moe_style_ == 2 && use_moe? partial_moe_output_ : input_residual,
                                                   layer_weight->ffn_weights.output_weight.bias,
                                                   layer_weight->post_ffn_layernorm_weights.gamma,
                                                   layer_weight->post_ffn_layernorm_weights.beta,
                                                   params_.layernorm_eps_,
                                                   h_token_num,
                                                   hidden_units,
                                                   nullptr,
                                                   nullptr,
                                                   stream_);

	    print_bsd(l, "decoder output", decoder_output, 1, h_token_num, hidden_units);

        sync_check_cuda_error();
        POP_RANGE;

        if (isLastLayerParallelId(l) == true && (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1)) {
            const int data_size = h_token_num * hidden_units / tensor_para_.world_size_;
            ftNcclSend(decoder_output + data_size * tensor_para_.rank_,
                       data_size,
                       pipeline_para_.rank_ + 1,
                       pipeline_para_,
                       stream_);
        }

        POP_RANGE;
    }
    POP_RANGE;

    // PUSH_RANGE(stream_, "Rebuild padding");
    // cudaD2Dcpy(base_ptr, decoder_layer_output_, h_token_num * hidden_units);
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    final_check_error();
}

template<typename T>
void ParallelGpt<T>::Ffnforward(std::unique_ptr<ExpertAttentionUtil<T>>& expert_attention_util,
        T* ffn_input_ptr, T* ffn_output_ptr, const size_t h_token_num, const DataType activation_in_type, const size_t hidden_units,
        const TensorMap* input_tensors, uint l, const size_t total_batch_size, const int* lora_input_lengths, const DataType activation_out_type,
        const int ffn_batch_size_lora, ParallelGptDecoderLayerWeight<T>* layer_weight, const bool use_moe, const bool vision) {
    size_t token_length = h_token_num;
    bool use_moe_instead_ffn = params_.moe_style_ == 1;
    // for cogvlm2, we perform expertFfn when there exists vision tokens in the input(in context stage), otherwise we perform expertFfn directly
    if (expert_attention_util && expert_attention_util->vision_token_length() > 0) {
        FT_CHECK(params_.moe_style_ == 0 && !use_moe && !use_moe_instead_ffn);
        if (vision) {
            ffn_input_ptr = expert_attention_util->vision_split_buf();
            ffn_output_ptr = expert_attention_util->vision_intermediate_buf();
            token_length = expert_attention_util->vision_token_length();
        } else {
            expert_attention_util->updateBufferShape(h_token_num, hidden_units, hidden_units);
            expert_attention_util->allocateBuffer();
            expert_attention_util->split();
            ffn_input_ptr = expert_attention_util->text_split_buf();
            ffn_output_ptr = expert_attention_util->text_intermediate_buf();
            token_length = expert_attention_util->text_token_length();
        }
    }
    TensorMap ffn_input_tensors(
        {{"ffn_input", Tensor{MEMORY_GPU, activation_in_type, {token_length, hidden_units}, ffn_input_ptr}},
        {"layer_id", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &l}},
        {"lora_ids", input_tensors->at("lora_ids")},
        {"lora_input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {total_batch_size}, lora_input_lengths}},
        {"batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {(size_t)1}, &ffn_batch_size_lora}}});
    if(quant_algo_.smoothQuantInt8()){
        FT_CHECK_WITH_INFO(ffn_intermediate_dynamic_scale_ != nullptr, "ffn_dynamic_scale should not be nullptr");
        ffn_input_tensors.insert("ffn_dynamic_scale", Tensor{MEMORY_GPU, TYPE_FP32, {token_length, 1}, ffn_intermediate_dynamic_scale_});
    }
    TensorMap ffn_output_tensors;
    size_t    moe_k = params_.moe_k_;
    ffn_output_tensors.insert("ffn_output",
                                Tensor{MEMORY_GPU, activation_out_type, {token_length, hidden_units}, ffn_output_ptr});
    if (use_moe_instead_ffn) {
        ffn_output_tensors.insert(
            "fc2_result",
            Tensor{MEMORY_GPU, activation_out_type, {moe_k * token_length, hidden_units}, fc2_result_});
        ffn_output_tensors.insert("expert_scales",
                                    Tensor{MEMORY_GPU, activation_out_type, {token_length, moe_k}, expert_scales_});
        ffn_output_tensors.insert(
            "expanded_source_row_to_expanded_dest_row",
            Tensor{MEMORY_GPU, TYPE_INT32, {token_length, moe_k}, expanded_source_row_to_expanded_dest_row_});
        ffn_output_tensors.insert("expert_for_source_row",
                                    Tensor{MEMORY_GPU, TYPE_INT32, {token_length, moe_k}, expert_for_source_row_});
    }

    ffn_layer_->forward(&ffn_output_tensors,
        &ffn_input_tensors,
        use_moe_instead_ffn ? &layer_weight->partial_moe_weights:
            vision ? &layer_weight->vision_ffn_weights: &layer_weight->ffn_weights,
        use_moe_instead_ffn);

    print_bsd(l, "post ffn", ffn_output_ptr, 1, token_length, hidden_units);

    if (params_.moe_style_ == 2 && use_moe) {
        print_bsd(l, "before moe", params_.layernorm_type_ == LayerNormType::pre_layernorm ? normed_self_attn_output_ : self_attn_output_, 1, h_token_num, hidden_units);

        TensorMap partial_moe_tensors;
        partial_moe_tensors.insert("ffn_output",
                Tensor{MEMORY_GPU, activation_out_type, {token_length, hidden_units}, partial_moe_output_});
        partial_moe_tensors.insert(
            "fc2_result",
            Tensor{MEMORY_GPU, activation_out_type, {moe_k * token_length, hidden_units}, fc2_result_});
        partial_moe_tensors.insert("expert_scales",
                                    Tensor{MEMORY_GPU, activation_out_type, {token_length, moe_k}, expert_scales_});
        partial_moe_tensors.insert(
            "expanded_source_row_to_expanded_dest_row",
            Tensor{MEMORY_GPU, TYPE_INT32, {token_length, moe_k}, expanded_source_row_to_expanded_dest_row_});
        partial_moe_tensors.insert("expert_for_source_row",
                                    Tensor{MEMORY_GPU, TYPE_INT32, {token_length, moe_k}, expert_for_source_row_});

        ffn_layer_->forward(&partial_moe_tensors, &ffn_input_tensors, &layer_weight->partial_moe_weights, true);

        print_bsd(l, "after partial moe", partial_moe_output_, 1, token_length, hidden_units);
    }
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;
#ifdef ENABLE_BF16
template class ParallelGpt<__nv_bfloat16>;
#endif

}  // namespace fastertransformer

#pragma once

#include <torch/torch.h>

#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/torch_impl/GptModel.hpp"

#define private public
#include "maga_transformer/cpp/models/GptModel.h"

using namespace std;
using namespace rtp_llm;
using namespace fastertransformer;

template <typename TestT>
class AttentionLayerTest : public DeviceTestBase {
public:
    using TestType = TestT;
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

    void testAttentionLayer(const CacheConfig& cache_conf,
                            const AttentionConfigs& attention_conf,
                            const std::vector<int32_t>& input_lengths,
                            const std::vector<int32_t>& sequence_lengths);

    AttentionLayerWeights getAttentionWeights(const GptAttention& gpt_attention);


};

torch::Tensor fakeAttentionInputs(const int64_t hidden_size, const int64_t token_num) {
    return torch::rand({token_num, hidden_size});
}

template <typename T>
AttentionLayerWeights AttentionLayerTest<T>::getAttentionWeights(const GptAttention& gpt_attention) {
    AttentionLayerWeights attention_weights;
    auto qkv_tensor = torch::concat({
        gpt_attention->q_proj->weight.transpose(0, 1),
        gpt_attention->k_proj->weight.transpose(0, 1),
        gpt_attention->v_proj->weight.transpose(0, 1)
    }, 1).to(dataTypeToTorchType(getTensorType<TestType>()));
    auto qkv_buf = tensorToBuffer(qkv_tensor);
    attention_weights.qkv_weight.reset(new DenseWeights(qkv_buf));
    auto o_buf = tensorToBuffer(gpt_attention->o_proj->weight.transpose(0, 1).contiguous().to(
        dataTypeToTorchType(getTensorType<TestType>())));
    attention_weights.output_weight.reset(new DenseWeights(o_buf));
    return attention_weights;
}

template <typename T>
void AttentionLayerTest<T>::testAttentionLayer(
    const CacheConfig& cache_conf,
    const AttentionConfigs& attention_conf,
    const std::vector<int32_t>& input_lengths,
    const std::vector<int32_t>& sequence_lengths)
{
    GptModelDescription description;
    Weights weights;
    description.attention_conf = attention_conf;
    GptModel model({device_, weights, description});
    auto dtype = getTensorType<TestType>();
    // 1. prepare inputs
    const auto context_token_num = std::accumulate(
        input_lengths.begin() + sequence_lengths.size(), input_lengths.end(), 0);
    const auto total_token_num = context_token_num + sequence_lengths.size();

    // NOTE: this relationship does not hold for all models, e.g. gemma
    const auto hidden_size = attention_conf.head_num * attention_conf.size_per_head;
    const auto input_tensor = fakeAttentionInputs(hidden_size, total_token_num);
    const auto context_lengths = std::vector<int32_t>(
        input_lengths.begin() + sequence_lengths.size(), input_lengths.end());
    const auto prefix_lengths = std::vector<int32_t>(context_lengths.size(), 0);

    const auto mask_tensor = create_context_mask(context_lengths, attention_conf.mask_type == AttentionMaskType::causalMask)
        .to(dataTypeToTorchType(dtype));
    std::cout << "mask: " << mask_tensor << std::endl;
    const auto input_buffer = tensorToBuffer(
        input_tensor.to(dataTypeToTorchType(dtype)));

    GptModelInputs model_inputs;
    model_inputs.combo_tokens = device_->clone({*tensorToBuffer(input_tensor)});
    model_inputs.input_lengths = device_->clone({*vector2Buffer(input_lengths), AllocationType::HOST});
    model_inputs.prefix_lengths = device_->clone({*vector2Buffer(prefix_lengths), AllocationType::HOST});
    model_inputs.sequence_lengths = device_->clone({*vector2Buffer(sequence_lengths), AllocationType::HOST});
    auto kv_cache = torch::empty(0);
    model_inputs.kv_cache_offset = allocateKVBlocks(cache_conf, input_lengths, kv_cache);
    auto kv_cache_buffer = cache_manager_->kvCacheBuffer();
    model_inputs.k_cache_buffer = kv_cache_buffer.k_blocks;
    model_inputs.v_cache_buffer = kv_cache_buffer.v_blocks;

    auto input_lengths_device = device_->clone({*model_inputs.input_lengths});
    auto sequence_lengths_device = device_->clone({*model_inputs.sequence_lengths});
    AttentionCommonInputs common_inputs({
            *input_lengths_device,
            *sequence_lengths_device
        });

    model.prepareAttentionInputs(model_inputs, dtype, common_inputs);

    auto layer_k_cache_buffer = model_inputs.k_cache_buffer->index(0);
    auto layer_v_cache_buffer = model_inputs.v_cache_buffer->index(0);
    common_inputs.kv_cache = KvCacheInfo({
        model_inputs.kv_cache_offset,
        layer_k_cache_buffer,
        layer_v_cache_buffer,
    });

    printBufferData(*model_inputs.kv_cache_offset, "kv_cache_offset");

    // 2. compute reference implementation result
    GptAttention gpt_attention(attention_conf);
    torch::nn::init::normal_(gpt_attention->q_proj->weight);
    torch::nn::init::normal_(gpt_attention->k_proj->weight);
    torch::nn::init::normal_(gpt_attention->v_proj->weight);
    torch::nn::init::normal_(gpt_attention->o_proj->weight);
    torch::nn::init::zeros_(gpt_attention->q_proj->bias);
    torch::nn::init::zeros_(gpt_attention->k_proj->bias);
    torch::nn::init::zeros_(gpt_attention->v_proj->bias);
    torch::nn::init::zeros_(gpt_attention->o_proj->bias);

    auto position_ids = create_position_ids(input_lengths);
    auto torch_output = gpt_attention->forward(input_tensor.unsqueeze(0), mask_tensor, position_ids)
                                     .reshape({-1, (int64_t)hidden_size});
    std::cout << "torch output: " << torch_output.sizes() << std::endl;

    // 3. compute kernel result and compare
    auto attention_weights = getAttentionWeights(gpt_attention);
    AttentionLayerParams params {
        *input_buffer,
        nullptr,
        attention_conf,
        attention_weights,
        common_inputs
    };
    auto attn_output = device_->attentionLayer(params);
    auto output_tensor = bufferToTensor(*attn_output.hidden_states);
    assertTensorClose(output_tensor, torch_output, 1e-3, 2);
}

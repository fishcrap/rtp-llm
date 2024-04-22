#pragma once
#include <assert.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <vector>
#include <map>
#include <torch/torch.h>
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/ptuning/Ptuning.h"
#include "maga_transformer/cpp/engines/EngineBase.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "src/fastertransformer/devices/utils/BufferTorchUtils.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class PtuningConstructor {
public:
    static std::unordered_map<int, PrefixParams> construct(const GptInitParameter& params, EngineBase* engine, CacheManager* cache_manager);
    static PrefixParams createPtuningV2(const GptInitParameter& params, CacheManager* cache_manager, torch::Tensor& prefix_prompt);

private:
    static void setKVPrefixBlock(const GptInitParameter& params, CacheManager* cache_manager,
                                    torch::Tensor& kv_prefix_prompt, std::vector<int>& prefix_block_indice);
    static std::unordered_map<int, PrefixParams> createMultiTaskPrompt(
        std::map<int, std::vector<int>> multi_task_prompt_tokens, EngineBase* engine, CacheManager* cache_manager);
};

} // namespace rtp_llm

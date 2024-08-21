#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "maga_transformer/cpp/lora/LoraManager.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase() {}

    static void initDevices(const EngineInitParams& params);
    ft::DeviceBase* getDevice() {
        return device_;
    }

    void addLora(int64_t lora_id, ft::lora::loraLayerWeightsMap lora_a, ft::lora::loraLayerWeightsMap lora_b);

    void removeLora(int64_t lora_id);

    std::shared_ptr<lora::LoraManager> getLoraManager();

    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;

    virtual absl::Status stop() = 0;

    virtual KVCacheInfo getKVCacheInfo() const {
        return {0, 0};
    }
protected:
    ft::DeviceBase* device_;
    std::shared_ptr<lora::LoraManager> lora_manager_;
};

}  // namespace rtp_llm

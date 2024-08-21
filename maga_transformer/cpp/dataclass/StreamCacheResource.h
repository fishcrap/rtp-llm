#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/BatchKVCacheBlockAddr.h"
#include <memory>

namespace rtp_llm {

class GenerateStream;

struct ResourceContext {
    std::shared_ptr<CacheManager>   cache_manager;
    std::shared_ptr<SystemPrompt>   system_prompt;
    bool                            reuse_cache{false};
};

class StreamCacheResource {
public:
    StreamCacheResource(
            GenerateStream* stream,
            const ResourceContext& resource_context,
            bool need_release_resource = true):
        stream_(stream),
        resource_context_(resource_context),
        need_release_resource_(need_release_resource) {}
    ~StreamCacheResource() {
        releaseResource();
    }
    void init(int batch_size);
    absl::StatusOr<int> initKVBlock(int token_capacity);
    absl::StatusOr<int> incrKVBlock(int token_capacity);
    int  tryReleaseKVBlock(size_t nums);
    void freeBatchBlocks(size_t batch_id, std::vector<int>& blocks);
    void releaseResource();
    int  singleBatchNeedBlocks(int seq_len) const;
    int  maxBlockSize() const;

    const BatchKVCacheBlockAddr& kvCache() const;
    void                         setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr);

    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    int seqSizePerBlock() const {
        return resource_context_.cache_manager->cacheConfig().seq_size_per_block;
    }

private:
    BatchKVCacheBlockAddr           batch_block_addr_;
    GenerateStream*                 stream_;
    ResourceContext                 resource_context_;
    int                             seq_size_per_block_    = 0;
    bool                            need_release_resource_ = true;
};

}  // namespace rtp_llm

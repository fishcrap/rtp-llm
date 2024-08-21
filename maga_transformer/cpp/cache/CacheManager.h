#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>
#include <thread>

#include "maga_transformer/cpp/cache/BlockCache.h"
#include "maga_transformer/cpp/cache/BlockRefCounter.h"
#include "maga_transformer/cpp/cache/CacheConfig.h"
#include "maga_transformer/cpp/cache/KVCacheBlockAddr.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "kmonitor/client/MetricsReporter.h"

namespace ft = fastertransformer;
namespace rtp_llm {

struct KVCacheInfo {
    size_t available_kv_cache;
    size_t total_kv_cache;
};


class CacheManager {
public:
    struct SeqPosition {
        int index;
        int offset;
    };

    struct KVCacheBuffer {
        ft::BufferPtr k_blocks;
        ft::BufferPtr v_blocks;
        ft::BufferPtr k_scale;
        ft::BufferPtr v_scale;
    };

    struct MatchInfo {
        std::vector<int> cache_blocks;
        int cache_block_num;
        int reuse_length;
        int reuse_block_num;
    };

public:
    CacheManager(const CacheConfig& config, ft::DeviceBase* device,
                 const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);
    ~CacheManager();

    const CacheConfig&     cacheConfig() const;
    const BlockRefCounter& blockRefCounter() const;
    const BlockCache&      blockCache() const;
    size_t                 freeBlockNums() const;
    size_t                 availableBlockNums() const;
    KVCacheInfo            getKVCacheInfo() const;
    size_t                 cacheItemNum() const;
    const KVCacheBuffer&   kvCacheBuffer() const;

    std::tuple<bool, KVCacheBlockAddr>      malloc(int nums = 1);
    std::tuple<bool, KVCacheBlockAddr, int> mallocWithCache(int want_block_nums, const std::vector<int>& token_ids);
    int                                     match(const std::vector<int>& token_ids);
    void                                    reserveBlocks(int nums);
    void                                    incrBlockRefCounter(const std::vector<int>& indices);

    void free(const std::vector<KVCacheBlockAddr>& resource);
    void free(const std::vector<int>& indice);
    void freeWithCache(const std::vector<int>& block_indices, const std::vector<int>& token_ids);
    void insertResidentCache(const std::vector<int>& block_indices, const std::vector<int>& token_ids);

    // TODO(xinfei.sxf) fix this two index?
    void setKVBlockValue(int kindex, int vindex, ft::BufferPtr& k_value, ft::BufferPtr& v_value);
    std::tuple<ft::BufferPtr, ft::BufferPtr> getKVBlockValue(int block_index);
    void blockCopy(int src_block_index, int dest_block_index);

    void reportMetricsLoop();

private:
    void                                    initFreeBlock();
    ft::BufferPtr                           tryAllocateMaxBuffer();
    void                                    allocateAndTpSync();
    void                                    initKvCache();
    MatchInfo                               matchImpl(const std::vector<int>& token_ids);
    std::tuple<bool, std::vector<int>>      mallocIndex(int nums = 1);
    std::tuple<bool, std::vector<int>>      mallocImpl(int nums);
    std::tuple<bool, std::vector<int>, int> mallocWithCacheImpl(int want_block_nums, const std::vector<int>& token_ids);
    void                                    maybeFreeBlockFromCache(int nums);

    void freeImpl(const std::vector<int>& indice);
    void insertIntoCache(const std::vector<int>& block_indices,
                         const std::vector<int>&              token_ids,
                         bool                                 is_resident);

    void copyKvCacheFromSeqIdxs(const std::vector<int>& block_indice_list, const std::vector<int>& src_index, const std::vector<int>& tgt_index);
    SeqPosition getSeqPosition(const std::vector<int>& block_indice_list, int idx);
    void        copyKvCacheFromSeqPosition(const SeqPosition& src_seq_position, const SeqPosition& dst_seq_position);

    void incrQueryRefCounter(const std::vector<int>& blocks);
    void decrQueryRefCounter(const std::vector<int>& blocks);

private:
    CacheConfig     config_;
    int             seq_size_per_block_;
    std::set<int>   free_blocks_index_;
    BlockRefCounter block_ref_counter_;
    BlockRefCounter query_ref_counter_;
    int             available_blocks_;
    BlockCache      block_cache_;
    KVCacheBuffer   kv_cache_;
    ft::DeviceBase* device_;
    bool            stop_ = false;
    std::thread     metrics_reporter_thread_;
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;

    // tmp
    ft::BufferPtr cache_aligned_buffer_;
    void*         cache_base_ptr_;
};

typedef std::shared_ptr<CacheManager> CacheManagerPtr;

}  // namespace rtp_llm

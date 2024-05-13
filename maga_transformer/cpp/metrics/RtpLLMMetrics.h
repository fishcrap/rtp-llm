#pragma once

#include "autil/Log.h"
#include "kmonitor/client/MetricsReporter.h"

namespace kmonitor {
class MetricsTags;
class MutableMetric;
}  // namespace kmonitor

namespace rtp_llm {

class RtpLLMStreamMetricsCollector final {
public:
    bool qps        = false;
    bool cancel_qps = false;
    bool error_qps  = false;

    int64_t total_latency_us       = 0;
    int64_t first_token_latency_us = 0;
    int64_t wait_latency_us        = 0;
    int64_t iterate_cout           = 0;
    int64_t reuse_length           = 0;
    int64_t input_token_length     = 0;
    int64_t output_token_length    = 0;
    int64_t query_batch_size       = 0;
};

class RtpLLMStreamMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector);

public:
    kmonitor::MutableMetric* qps_metric                    = nullptr;
    kmonitor::MutableMetric* cancel_qps_metric             = nullptr;
    kmonitor::MutableMetric* error_qps_metric              = nullptr;
    kmonitor::MutableMetric* total_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* first_token_latency_us_metric = nullptr;
    kmonitor::MutableMetric* wait_latency_us_metric        = nullptr;
    kmonitor::MutableMetric* iterate_cout_metric           = nullptr;
    kmonitor::MutableMetric* reuse_length_metric           = nullptr;
    kmonitor::MutableMetric* input_token_length_metric     = nullptr;
    kmonitor::MutableMetric* output_token_length_metric    = nullptr;
    kmonitor::MutableMetric* query_batch_size_metric       = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMSchedulerMetricsCollector final {
public:
    int64_t fallback_stream_size = 0;
    int64_t wait_stream_size     = 0;
    int64_t running_stream_size  = 0;
};

class RtpLLMSchedulerMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector);

public:
    kmonitor::MutableMetric* fallback_stream_size_metric = nullptr;
    kmonitor::MutableMetric* wait_stream_size_metric     = nullptr;
    kmonitor::MutableMetric* running_stream_size_metric  = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMEngineMetricsCollector final {
public:
    bool    update_lora_qps       = false;
    bool    error_update_lora_qps = false;
    int64_t step_latency_us       = 0;
};

class RtpLLMEngineMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMEngineMetricsCollector* collector);

public:
    kmonitor::MutableMetric* step_latency_us_metric       = nullptr;
    kmonitor::MutableMetric* update_lora_qps_metric       = nullptr;
    kmonitor::MutableMetric* error_update_lora_qps_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMExecutorMetricsCollector final {
public:
    int64_t context_batch_size  = 0;
    int64_t generate_batch_size = 0;
    int64_t execute_token_size  = 0;
    int64_t max_seq_len         = 0;
};

class RtpLLMExecutorMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector);

public:
    kmonitor::MutableMetric* context_batch_size_metric  = nullptr;
    kmonitor::MutableMetric* generate_batch_size_metric = nullptr;
    kmonitor::MutableMetric* execute_token_size_metric  = nullptr;
    kmonitor::MutableMetric* max_seq_len                = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

class RtpLLMCacheMetricsCollector final {
public:
    int64_t kv_cache_item_num = 0;
    int64_t kv_cache_left_seq = 0;
};

class RtpLLMCacheMetrics: public kmonitor::MetricsGroup {
public:
    bool init(kmonitor::MetricsGroupManager* manager) override;
    void report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector);

public:
    kmonitor::MutableMetric* kv_cache_item_num_metric = nullptr;
    kmonitor::MutableMetric* kv_cache_left_seq_metric = nullptr;

private:
    AUTIL_LOG_DECLARE();
};

bool initKmonitorFactory(const std::string& tenant, const std::string& sink_address);

kmonitor::MetricsTags getHippoTags();

}  // namespace rtp_llm

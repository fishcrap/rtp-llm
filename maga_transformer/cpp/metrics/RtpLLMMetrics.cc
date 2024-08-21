#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "src/fastertransformer/utils/logger.h"
#include "kmonitor/client/KMonitorFactory.h"
#include "maga_transformer/cpp/metrics/KmonParam.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RtpLLMStreamMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpEmbeddingGlobalMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpEmbeddingStreamMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMSchedulerMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMCacheMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMCacheReuseMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMExecutorMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMTokenPSMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMEngineMetrics);
AUTIL_LOG_SETUP(rtp_llm, RtpLLMKernelMetrics);

#define REPORT_QPS(name)                                                                                               \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_QPS(name##_metric);                                                                             \
    }

#define REPORT_GAUGE(name)                                                                                             \
    if (collector->name) {                                                                                             \
        REPORT_MUTABLE_METRIC(name##_metric, collector->name);                                                         \
    }

bool RtpLLMStreamMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "rtp_llm_framework_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "rtp_llm_framework_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(cancel_qps_metric, "rtp_llm_cancel_qps");

    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "rtp_llm_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(first_token_latency_us_metric, "rtp_llm_first_token_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_latency_us_metric, "rtp_llm_wait_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(pause_latency_us_metric, "rtp_llm_pause_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(iterate_count_metric, "rtp_llm_iterate_count");
    REGISTER_GAUGE_MUTABLE_METRIC(reuse_length_metric, "rtp_llm_reuse_length");
    REGISTER_GAUGE_MUTABLE_METRIC(input_token_length_metric, "rtp_llm_input_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(output_token_length_metric, "rtp_llm_output_token_length");
    REGISTER_GAUGE_MUTABLE_METRIC(query_batch_size_metric, "rtp_llm_query_batch_size");

    REGISTER_GAUGE_MUTABLE_METRIC(fallback_tokens_metric, "rtp_llm_fallback_tokens");
    REGISTER_GAUGE_MUTABLE_METRIC(fallback_times_metric, "rtp_llm_fallback_times");

    return true;
}

void RtpLLMStreamMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMStreamMetricsCollector* collector) {
    REPORT_QPS(qps);
    REPORT_QPS(cancel_qps);
    REPORT_QPS(error_qps);

    REPORT_GAUGE(total_latency_us);
    REPORT_GAUGE(first_token_latency_us);
    REPORT_GAUGE(wait_latency_us);
    REPORT_GAUGE(pause_latency_us);
    REPORT_GAUGE(iterate_count);
    REPORT_GAUGE(reuse_length);
    REPORT_GAUGE(input_token_length);
    REPORT_GAUGE(output_token_length);
    REPORT_GAUGE(query_batch_size);

    REPORT_GAUGE(fallback_tokens);
    REPORT_GAUGE(fallback_times);
}

// for rpc request
bool RtpEmbeddingGlobalMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "py_rtp_framework_rt");
    REGISTER_QPS_MUTABLE_METRIC(error_qps_metric, "py_rtp_framework_error_qps");
    REGISTER_QPS_MUTABLE_METRIC(qps_metric, "py_rtp_framework_qps");
    REGISTER_QPS_MUTABLE_METRIC(success_qps_metric, "py_rtp_success_qps_metric");
    return true;
}

void RtpEmbeddingGlobalMetrics::report(const kmonitor::MetricsTags* tags, RtpEmbeddingGlobalMetricsCollector* collector) {
    REPORT_MUTABLE_QPS(qps_metric);
    if (collector->error) {
        REPORT_MUTABLE_QPS(error_qps_metric);
    } else {
        REPORT_MUTABLE_QPS(success_qps_metric);
        REPORT_GAUGE(total_latency_us);
    }
}

bool RtpEmbeddingStreamMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(total_latency_us_metric, "rtp_llm_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_latency_us_metric, "rtp_llm_wait_latency_us");
    REGISTER_GAUGE_MUTABLE_METRIC(input_token_length_metric, "rtp_llm_input_token_length");
    return true;
}

void RtpEmbeddingStreamMetrics::report(const kmonitor::MetricsTags* tags, RtpEmbeddingStreamMetricsCollector* collector) {
    REPORT_GAUGE(total_latency_us);
    REPORT_GAUGE(wait_latency_us);
    REPORT_GAUGE(input_token_length);
}

bool RtpLLMSchedulerMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(fallback_stream_size_metric, "rtp_llm_fallback_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(wait_stream_size_metric, "rtp_llm_wait_stream_size");
    REGISTER_GAUGE_MUTABLE_METRIC(running_stream_size_metric, "rtp_llm_running_stream_size");
    return true;
}

void RtpLLMSchedulerMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMSchedulerMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(fallback_stream_size_metric, collector->fallback_stream_size);
    REPORT_MUTABLE_METRIC(wait_stream_size_metric, collector->wait_stream_size);
    REPORT_MUTABLE_METRIC(running_stream_size_metric, collector->running_stream_size);
}

bool RtpLLMEngineMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_QPS_MUTABLE_METRIC(update_lora_qps_metric, "rtp_llm_update_lora_qps");
    REGISTER_QPS_MUTABLE_METRIC(error_update_lora_qps_metric, "rtp_llm_error_update_lora_qps");

    REGISTER_GAUGE_MUTABLE_METRIC(step_latency_us_metric, "rtp_llm_step_latency_us");
    return true;
}

void RtpLLMEngineMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMEngineMetricsCollector* collector) {
    REPORT_QPS(update_lora_qps);
    REPORT_QPS(error_update_lora_qps);

    REPORT_GAUGE(step_latency_us);
}

bool RtpLLMExecutorMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(context_batch_size_metric, "rtp_llm_context_batch_size");
    REGISTER_GAUGE_MUTABLE_METRIC(generate_batch_size_metric, "rtp_llm_generate_batch_size");
    REGISTER_GAUGE_MUTABLE_METRIC(execute_token_size_metric, "rtp_llm_execute_token_size");
    REGISTER_GAUGE_MUTABLE_METRIC(max_seq_len, "rtp_llm_max_seq_len");
    return true;
}

void RtpLLMExecutorMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMExecutorMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(context_batch_size_metric, collector->context_batch_size);
    REPORT_MUTABLE_METRIC(generate_batch_size_metric, collector->generate_batch_size);
    REPORT_MUTABLE_METRIC(execute_token_size_metric, collector->execute_token_size);
    REPORT_MUTABLE_METRIC(max_seq_len, collector->max_seq_len);
}

bool RtpLLMTokenPSMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(context_tps_metric, "rtp_llm_context_tps");
    REGISTER_GAUGE_MUTABLE_METRIC(generate_tps_metric, "rtp_llm_generate_tps");
    REGISTER_GAUGE_MUTABLE_METRIC(total_tps_metric, "rtp_llm_total_tps");
    return true;
}

void RtpLLMTokenPSMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMTokenPSMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(context_tps_metric, collector->context_tps);
    REPORT_MUTABLE_METRIC(generate_tps_metric, collector->generate_tps);
    REPORT_MUTABLE_METRIC(total_tps_metric, collector->total_tps);
}

bool RtpLLMCacheMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_item_num_metric, "rtp_llm_kv_cache_item_num");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_left_seq_metric, "rtp_llm_kv_cache_left_seq");
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_used_ratio_metric, "rtp_llm_kv_cache_used_ratio");
    return true;
}

void RtpLLMCacheMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMCacheMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kv_cache_item_num_metric, collector->kv_cache_item_num);
    REPORT_MUTABLE_METRIC(kv_cache_left_seq_metric, collector->kv_cache_left_seq);
    REPORT_MUTABLE_METRIC(kv_cache_used_ratio_metric, collector->kv_cache_used_ratio);
}

bool RtpLLMCacheReuseMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kv_cache_reuse_length, "rtp_llm_kv_cache_reuse_length");
    return true;
}

void RtpLLMCacheReuseMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMCacheReuseMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kv_cache_reuse_length, collector->kv_cache_reuse_length);
}

bool RtpLLMKernelMetrics::init(kmonitor::MetricsGroupManager* manager) {
    REGISTER_GAUGE_MUTABLE_METRIC(kernel_exec_time_metric, "rtp_llm_kenrel_exec_time");
    return true;
}

void RtpLLMKernelMetrics::report(const kmonitor::MetricsTags* tags, RtpLLMKernelMetricsCollector* collector) {
    REPORT_MUTABLE_METRIC(kernel_exec_time_metric, collector->kernel_exec_time);
}

#undef REPORT_QPS
#undef REPORT_GAUGE

std::string getEnvWithDefault(const std::string& name, const std::string& default_value) {
    if (std::getenv(name.c_str())) {
        return std::getenv(name.c_str());
    } else {
        return default_value;
    }
}

bool initKmonitorFactory() {
    KmonParam param;
    param.init();
    if (!param.kmonitorMetricsReporterCacheLimit.empty()) {
        size_t limit = 0;
        if (autil::StringUtil::fromString<size_t>(param.kmonitorMetricsReporterCacheLimit, limit) || limit > 0) {
            kmonitor::MetricsReporter::setMetricsReporterCacheLimit(limit);
            FT_LOG_INFO("set metrics reporter cache limit [%lu].", limit);
        }
    }

    if (param.kmonitorNormalSamplePeriod > 0) {
        FT_LOG_INFO("set kmonitor normal sample period [%d] seconds.", param.kmonitorNormalSamplePeriod);
        kmonitor::MetricLevelConfig config;
        config.period[kmonitor::NORMAL] = (unsigned int)param.kmonitorNormalSamplePeriod;
        kmonitor::MetricLevelManager::SetGlobalLevelConfig(config);
    }

    kmonitor::MetricsConfig metricsConfig;
    metricsConfig.set_tenant_name(param.kmonitorTenant);
    metricsConfig.set_service_name(param.kmonitorServiceName);
    metricsConfig.set_sink_address((param.kmonitorSinkAddress + ":" + param.kmonitorPort).c_str());
    metricsConfig.set_enable_log_file_sink(param.kmonitorEnableLogFileSink);
    metricsConfig.set_enable_prometheus_sink(param.kmonitorEnablePrometheusSink);
    metricsConfig.set_manually_mode(param.kmonitorManuallyMode);
    metricsConfig.set_inited(true);
    metricsConfig.AddGlobalTag("hippo_slave_ip", param.hippoSlaveIp);
    for (auto &pair : param.kmonitorTags) {
        metricsConfig.AddGlobalTag(pair.first, pair.second);
    }
    if (!kmonitor::KMonitorFactory::Init(metricsConfig)) {
        FT_LOG_ERROR("init kmonitor factory failed with");
        return false;
    }
    FT_LOG_INFO("before KMonitorFactory::Start(), config");
    kmonitor::KMonitorFactory::Start();
    FT_LOG_INFO("KMonitorFactory::Start() finished");
    kmonitor::KMonitorFactory::registerBuildInMetrics(nullptr, param.kmonitorMetricsPrefix);
    FT_LOG_INFO("KMonitorFactory::registerBuildInMetrics() finished");
    return true;
}

kmonitor::MetricsTags getHippoTags() {
    auto hippo_tags = kmonitor::MetricsTags();
    if (std::getenv("HIPPO_ROLE")) {
        hippo_tags.AddTag("host_ip", getEnvWithDefault("HIPPO_SLAVE_IP", ""));
        hippo_tags.AddTag("container_ip", getEnvWithDefault("RequestedIP", ""));
        hippo_tags.AddTag("hippo_role", getEnvWithDefault("HIPPO_ROLE", ""));
        hippo_tags.AddTag("hippo_app", getEnvWithDefault("HIPPO_APP", ""));
        hippo_tags.AddTag("hippo_group", getEnvWithDefault("HIPPO_SERVICE_NAME", ""));
    }
    return hippo_tags;
}

}  // namespace rtp_llm

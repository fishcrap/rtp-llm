#include "maga_transformer/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/Weights.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams& params,
                               const std::shared_ptr<CacheManager>& cache_manager,
                               ft::DeviceBase* device,
                               const std::shared_ptr<lora::LoraManager>& lora_manager = nullptr):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_)),
    dtype_(ft::getDataType(params.gpt_init_parameter.data_type_)),
    is_causal_(params.gpt_init_parameter.is_causal_)
{
    int eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{device_, eos_id, 1024}; // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    model_.reset(new GptModel({device_, params.gpt_weights, genModelDescription(params.gpt_init_parameter)}));
    batch_stream_processor_.reset(new NormalBatchStreamProcessor(params.gpt_init_parameter));
}


absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups stream_groups(streams);
    reportMetrics(stream_groups);
    auto model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
    RETURN_IF_STATUS_OR_ERROR(model_input_status);
    auto& model_input = model_input_status.value();
    tpSyncModelInputs(model_input, device_);
    // get lora input
    if (lora_manager_ != nullptr) {
        model_input.lora_model_input = lora_manager_->makeLoraModelInput(model_input.lora_ids,
                                                                         model_input.lora_input_lengths);
    }
    auto kv_cache_buffer = cache_manager_->kvCacheBuffer();
    model_input.k_cache_buffer = kv_cache_buffer.k_blocks;
    model_input.v_cache_buffer = kv_cache_buffer.v_blocks;
    model_input.k_scale_buffer = kv_cache_buffer.k_scale;
    model_input.v_scale_buffer = kv_cache_buffer.v_scale;
    FT_LOG_DEBUG("model_input: %s", model_input.debugString().c_str());
    auto            merged_output = std::make_unique<MergedOutput>();
    GptModelOutputs model_output;
    model_output = std::move(model_->forward(model_input));
    FT_LOG_DEBUG("model forward done");
    if (device_->getDeviceProperties().tp_rank > 0) {
        return absl::OkStatus();
    }
    auto sampler_input_status = batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output);
    RETURN_IF_STATUS_OR_ERROR(sampler_input_status);
    auto& sampler_input           = sampler_input_status.value();
    merged_output->model_output   = std::move(model_output);
    merged_output->sampler_output = std::move(sampler_->forward(sampler_input));
    FT_LOG_DEBUG("sampler forward done");
    return batch_stream_processor_->dispatch(stream_groups, sampler_input, merged_output);
}

void NormalExecutor::reportMetrics(const StreamGroups& stream_groups) {
    if (metrics_reporter_) {
        RtpLLMExecutorMetricsCollector executor_collector;
        executor_collector.context_batch_size  = stream_groups.contextStreams().size();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        RtpLLMTokenPSMetricsCollector tps_collector;
        tps_collector.context_tps = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

}  // namespace rtp_llm

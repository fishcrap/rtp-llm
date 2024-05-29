#pragma once

#include "maga_transformer/cpp/deprecated/ParallelModelWrapper.h"
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"
#include "maga_transformer/cpp/embedding_engine/handlers/HandlerBase.h"

#include <memory>
namespace rtp_llm {

class EmbeddingExecutor {
public:
    explicit EmbeddingExecutor(const fastertransformer::GptInitParameter& gpt_init_parameter,
                               ft::NcclParam                              tensor_para,
                               ft::NcclParam                              pipeline_para,
                               const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layer_weights,
                               const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights,
                               py::object                                                              handler,
                               const kmonitor::MetricsReporterPtr metrics_reporter = nullptr);

    absl::Status process(const std::list<EmbeddingStreamPtr>& streams);

private:
    std::unique_ptr<ParallelModelWrapper> model_wrapper_;
    py::object                            handler_;
    ft::NcclParam                         tensor_para_;
    ft::NcclParam                         pipeline_para_;
    ft::DeviceBase*                       device_;
    ft::BufferPtr                         max_position_ids_buf_;
    ft::DataType                          data_type_;
    kmonitor::MetricsReporterPtr          metrics_reporter_ = nullptr;
    const fastertransformer::GptInitParameter& params_;    

    ModelRequest                     generateOldModelRequest(GptModelInputs& model_input);
    absl::StatusOr<GptModelInputs>   gatherModelInput(const std::list<EmbeddingStreamPtr>& streams) const;
    std::unique_ptr<GptModelOutputs> copyResultToCPU(th::Tensor gpu_outputs) const;
    absl::Status                     updateStreams(th::Tensor    gpu_outputs,
                                                   const std::list<EmbeddingStreamPtr>& streams) const;
    absl::Status                     createAttentionMask(GptModelInputs& model_input) const;
    absl::StatusOr<th::Tensor>       postProcess(const ModelRequest& model_request, const GptModelOutputs& gpu_outputs);
    void calcTokenNum(const std::list<EmbeddingStreamPtr>& streams, int64_t& token_num, int64_t& batch_size) const;    
    void                             init_position_ids(int max_seq_len);
    void reportMetrics(size_t context_batch_size, size_t combo_token_num, size_t max_seq_len) const;
};
}  // namespace rtp_llm
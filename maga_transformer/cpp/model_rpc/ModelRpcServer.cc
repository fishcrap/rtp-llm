#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/normal_engine/NormalEngine.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/proto/model_rpc_service.pb.h"
#include "autil/TimeUtility.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace std;

namespace rtp_llm {

// TODO: not use absl::status
int transErrorCode(absl::StatusCode code) {
    const static std::unordered_map<int, int> error_code_map = {
        {8, 602}, // kResourceExhausted, MALLOC_ERROR
        {4, 603}, // kDeadlineExceeded, TIMEOUT_ERROR
        {13, 514} // kInternal, UNKNOWN_ERROR
    };
    auto it = error_code_map.find((int)code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return 514;
    }
}

ModelRpcServiceImpl::ModelRpcServiceImpl(
    const EngineInitParams& maga_init_params) {
    engine_.reset(new NormalEngine(maga_init_params));
}

ModelRpcServiceImpl::ModelRpcServiceImpl(
    const EngineInitParams& maga_init_params, py::object mm_process_engine) {
    engine_.reset(new NormalEngine(maga_init_params));
    if (!mm_process_engine.is_none()) {
        mm_processor_.reset(new MultimodalProcessor(mm_process_engine,
            maga_init_params.gpt_init_parameter.mm_sep_tokens_,
            maga_init_params.gpt_init_parameter.include_sep_tokens_));
    }
}

grpc::Status ModelRpcServiceImpl::generate_stream(grpc::ServerContext*                  context,
                                                  const GenerateInputPB*                request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    FT_LOG_DEBUG("receive request %ld", request->request_id());
    auto input = QueryConverter::transQuery(request);

    // todo: catch python exception, such as download timeout
    if (mm_processor_ != nullptr && input->multimodal_urls) {
        auto mm_res = mm_processor_->update_mm_features(input);
        if (!mm_res.ok()) {
            return grpc::Status(grpc::StatusCode::CANCELLED, mm_res.ToString());
        }
    }
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->lora_id);
    FT_LOG_DEBUG("request:[%ld] trans to stream success", request->request_id());
    auto stream = engine_->enqueue(input);
    FT_LOG_DEBUG("request:[%ld] enqueue success", request->request_id());
    while (!stream->finished()) {
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        const auto output_status = stream->nextOutput();
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        if (!output_status.ok()) {
            FT_LOG_INFO("request:[%ld] generate error %s", request->request_id(), output_status.status().ToString().c_str());
            auto status = output_status.status();
            ErrorDetailsPB error_details;
            error_details.set_error_code(transErrorCode(status.code()));
            error_details.set_error_message(status.ToString());
            std::string error_details_serialized;
            if (error_details.SerializeToString(&error_details_serialized)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString(), error_details_serialized);
            } else {
                FT_LOG_INFO("request:[%ld] SerializeToString error", request->request_id());
                return grpc::Status(grpc::StatusCode::INTERNAL, status.ToString());
            }
        }
        FT_LOG_DEBUG("request:[%ld] generate next output success", request->request_id());
        GenerateOutputsPB outputs_pb;
        QueryConverter::transResponse(&outputs_pb, &(output_status.value()));
        if (context->IsCancelled()) {
            stream->cancel();
            FT_LOG_DEBUG("request:[%ld] cancel", request->request_id());
            break;
        }
        if (!writer->Write(outputs_pb)) {
            FT_LOG_INFO("request:[%ld] write outputs pb failed", request->request_id());
            stream->cancel();
            break;
        }
    }
    FT_LOG_DEBUG("request:[%ld] generate done", request->request_id());
    return grpc::Status::OK;
}


KVCacheInfo ModelRpcServiceImpl::getKVCacheInfo() const {
    return engine_->getKVCacheInfo();
}

void ModelRpcServiceImpl::addLora(const int64_t lora_id,
                                  const ft::lora::loraLayerWeightsMap& lora_a_weights,
                                  const ft::lora::loraLayerWeightsMap& lora_b_weights)
{
    engine_->addLora(lora_id, lora_a_weights, lora_b_weights);
}
void ModelRpcServiceImpl::removeLora(const int64_t lora_id) {
    engine_->removeLora(lora_id);
}

}  // namespace rtp_llm

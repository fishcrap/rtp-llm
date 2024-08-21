#pragma once

#include "grpc++/grpc++.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/model_rpc/ModelRpcServer.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

class RtpLLMOp: public th::jit::CustomClassHolder {
public:
    RtpLLMOp();
    ~RtpLLMOp();
    void init(const ft::GptInitParameter& gpt_init_parameter, py::object py_layer_weights,
              py::object py_weights, py::object mm_process_engine);

    void addLora(const int64_t lora_id, py::object lora_a_weights, py::object lora_b_weights);
    void removeLora(const int64_t lora_id);
    void stop();
    void _init(int64_t model_rpc_port, const rtp_llm::EngineInitParams maga_init_params, py::object mm_process_engine);
    std::tuple<int64_t, int64_t> getKVCacheInfo();
    // std::shared_ptr<rtp_llm::GenerateStream> forward(std::shared_ptr<rtp_llm::GenerateInput> query);

private:
    std::unique_ptr<rtp_llm::ModelRpcServiceImpl> model_rpc_server_;
    std::unique_ptr<grpc::Server>                 grpc_server_;
    std::thread                                   grpc_server_thread_;
    std::atomic<bool>                             is_server_ready_{false};
    std::atomic<bool>                             is_server_shutdown_{false};
};

void registerRtpLLMOp(const py::module& m);

}  // namespace torch_ext

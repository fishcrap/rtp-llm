#pragma once

#include "src/fastertransformer/devices/Weights.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeTypes.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/layernorm_types.h"

#include <optional>
#include <functional>
#include <sstream>
#include <memory>

namespace fastertransformer {

enum class OpErrorType {
    ERROR_NONE,
    ERROR_INVALID_ARGS,
    ERROR_RESOURCE_EXHAUSTED,
    ERROR_UNIMPLEMENTED,
    ERROR_INTERNAL,
    ERROR_UNKNOWN,
};

class OpStatus {
public:
    OpStatus(OpErrorType, const std::string& message = "")
    : error_type(OpErrorType::ERROR_NONE), error_message(message) {}

    static OpStatus make(OpErrorType error_type, const std::string& error_message = "") {
        return OpStatus(error_type, error_message);
    }
    static OpStatus OK() { return OpStatus(OpErrorType::ERROR_NONE); }

    bool ok() const { return error_type == OpErrorType::ERROR_NONE; }
public:
    OpErrorType error_type;
    std::string error_message;
};

class OpException : public std::exception {
public:
    OpException(const OpStatus& status)
    : status_(status) {}

    const char* what() const noexcept override {
        std::stringstream ss;
        ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message;
        return status_.error_message.c_str();
    }

    const OpStatus& status() const { return status_; }
private:
    OpStatus status_;
};

using OptionalConstBufferRef = std::optional<std::reference_wrapper<const Buffer>>;
using OptionalBufferRef = std::optional<std::reference_wrapper<Buffer>>;

struct CopyParams {
    CopyParams(const Buffer& src, Buffer& dst) : src(src), dst(dst) {}

    const Buffer& src;
    Buffer&       dst;

    const std::optional<std::vector<size_t>> src_offset;
    const std::optional<std::vector<size_t>> dst_offset;
    const std::optional<std::vector<size_t>> sizes;
};

struct LayernormOutput {
    BufferPtr norm_output;
    BufferPtr add_bias_output;
};

// The Layernorm Op has fused Layernorm and AddBias functionality
// if gamma and beta are not provided, output = input * alpha + residual1 + bias if alpha is provided;
// else output = input + residual1 + residual2 + bias
struct LayernormParams {

    // for layernorm
    LayernormParams(const NormType norm_type, const Buffer& input, OptionalBufferRef bias_output,
                    OptionalConstBufferRef residual1, OptionalConstBufferRef bias,
                    const LayerNormWeights& weights, double eps = 1e-6):
    norm_type(norm_type), input(input), bias_output(bias_output),
    residual1(residual1), bias(bias), weights(weights), eps(eps) {}

    const NormType norm_type = NormType::layernorm;
    const Buffer&  input;
    OptionalBufferRef bias_output;

    const OptionalConstBufferRef  residual1;
    const OptionalConstBufferRef  residual2;
    const OptionalConstBufferRef  bias;
    const std::optional<double>   alpha;

    const std::optional<std::reference_wrapper<const LayerNormWeights>> weights;
    const double eps;
};

// corresponds to cublasOperation_t
enum class TransposeOperation {
    NONE                = 0,
    TRANSPOSE           = 1,
    CONJUGATE_TRANSPOSE = 2,
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of A, B, C, D have two options: [m, k], [k, n], [m, n], [m, n]
// or [bs, m, k], [bs, k, n], [bs, m, n], [bs, m, n] where bs is batch_size
// NOTE: caller needs to preallocate C
// TODO: whether inplace ?
struct GemmParams {

    GemmParams(const Buffer& A,
               const Buffer& B,
               OptionalBufferRef C = std::nullopt,
               const DataType compute_type = DataType::TYPE_INVALID,
               TransposeOperation transA = TransposeOperation::NONE,
               TransposeOperation transB = TransposeOperation::NONE):
               A(A),
               B(B),
               C(C),
               compute_type(compute_type),
               transA(transA),
               transB(transB) {}


    const Buffer& A;
    const Buffer& B;
    OptionalBufferRef C;
    const DataType compute_type = DataType::TYPE_INVALID; // If passed invalid type, op should infer type

    const TransposeOperation transA = TransposeOperation::NONE;
    const TransposeOperation transB = TransposeOperation::NONE;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    void check() const;
};

struct GroupedGemmOutput {
    BufferPtr D;
};

// D = alpha * op(A) * op(B) + beta * C
// shapes of each A, B, C, D needs to be [m, k], [k, n], [m, n], [m, n]
struct GroupedGemmParams {
    using OutputType = GroupedGemmOutput;

    GroupedGemmParams(
        const std::vector<Buffer>& A,
        const std::vector<Buffer>& B,
        std::vector<Buffer>& C
    ) : A(A), B(B), C(C), D(C) {}
    GroupedGemmParams(
        const std::vector<Buffer>& A,
        const std::vector<Buffer>& B,
        const std::vector<Buffer>& C,
        std::vector<Buffer>&       D
    ) : A(A), B(B), C(C), D(D) {}

    const std::vector<Buffer>& A;
    const std::vector<Buffer>& B;
    const std::vector<Buffer>& C;
    std::vector<Buffer>&       D;
};

struct EmbeddingLookupParams {
    const Buffer& combo_tokens;
    const Buffer& embedding_table;

    const std::optional<std::reference_wrapper<const Buffer>> position_ids;
    const std::optional<std::reference_wrapper<const Buffer>> position_table;
};

struct AttentionCommonInputs {
    const Buffer& kv_cache_blocks; // [batch_size, block_length], int64 block pointers
    const std::optional<std::reference_wrapper<const Buffer>> kv_cache_scales;

    const Buffer& input_lengths;
    const Buffer& sequence_lengths;

    const std::optional<std::reference_wrapper<const Buffer>> padding_offset;
    const std::optional<std::reference_wrapper<const Buffer>> position_ids;
    const std::optional<std::reference_wrapper<const Buffer>> attention_mask;
    const std::optional<std::reference_wrapper<const Buffer>> linear_bias_slopes;
    const std::optional<std::reference_wrapper<const Buffer>> prefix_prompt_lengths;
    const std::optional<bool>         count_prefix_length;
    const std::optional<uint32_t>     max_prefix_length;

    const std::optional<std::reference_wrapper<const Buffer>> lora_ids;
    const std::optional<std::reference_wrapper<const Buffer>> lora_input_lengths;
};

// TODO(wangyin): figure out these styles and doc them.
enum class PositionEmbeddingStyle {
    BaseRotaryEmbedding          = 0,
    LinearScalar  = 1,
    NTKScalar     = 2,
    DynamicNTKS   = 3,
    GLM           = 4,
};

struct AttentionConfigs {
    PositionEmbeddingStyle position_embedding_style;
    int64_t rotary_embedding_dim      = 0;
    int64_t rotary_embedding_base     = 10000;
    double  dynamic_embedding_scalar  = 0.0;
    int64_t dynamic_embedding_max_pos = 0;
    int64_t position_embeddings_scale = 1;
    int64_t base_scale                = 1;

    bool    use_logn_attn = false;
    int64_t logn_seq_len  = 2048;
};

struct AttentionModuleOutput {
    BufferPtr hidden_states;
};

struct AttentionModuleParams {
    const Buffer& input;

    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;

    uint32_t batch_size;
    uint32_t max_seq_length;

    AttentionCommonInputs& common;
};

struct AttentionLayerOutput {
    BufferPtr hidden_states;
};

struct AttentionLayerParams {
    const Buffer& input;

    const AttentionConfigs&      configs;
    const AttentionLayerWeights& weights;
    AttentionCommonInputs& common;
};

struct FfnLayerOutput {
    BufferPtr hidden_states;
};

struct FfnLayerParams {
    FfnLayerParams(const Buffer& input,
                   const FfnLayerWeights& weights,
                   const ActivationType atype) :
                   input(input),
                   weights(weights),
                   activation_type(atype)
                   {}

    const Buffer& input;

    const FfnLayerWeights&       weights;
    const ActivationType         activation_type;

    const OptionalConstBufferRef lora_ids;
    const OptionalConstBufferRef lora_input_lengths;
};

struct SamplerConfig {
    const size_t         num_beams;
    const Buffer&        top_k;
    const Buffer&        top_p;
    const Buffer&        temperature;
    const Buffer&        random_seed;
    const Buffer&        repetition_penalty;
};

struct SamplerParams {
    const Buffer& logits;
    const Buffer& step;              // shape: [1]
    const Buffer& max_input_length;  // shape: [1]
    const Buffer& input_lengths;     // shape: [batch_size]
    const Buffer& ite;               // shape: [1]
    const Buffer& eos_id;
    Buffer& kv_cache_blocks;

    const SamplerConfig config;
};

struct SamplerOutput {
    BufferPtr output_ids;
    BufferPtr sequence_length;
    BufferPtr finished;
    BufferPtr cum_log_probs;
    BufferPtr output_log_probs;
};

struct BroadcastParams {
    std::vector<Buffer>& buffers;
    const int64_t        root;
};

struct AllReduceParams {
    std::vector<Buffer>& buffers;
};

// output = act(input) + bias
struct ActivationParams {
    ActivationType atype;
    Buffer& states;
    const OptionalConstBufferRef bias;
    const OptionalConstBufferRef gate;
    const OptionalConstBufferRef gate_bias;

    ActivationParams(ActivationType atype,
                     Buffer& states,
                     OptionalConstBufferRef bias = std::nullopt,
                     OptionalConstBufferRef gate = std::nullopt,
                     OptionalConstBufferRef gate_bias = std::nullopt) :
                     atype(atype),
                     states(states),
                     bias(bias),
                     gate(gate),
                     gate_bias(gate_bias) {}
};

struct SoftmaxParams{

    SoftmaxParams(const Buffer& input,
                  const Buffer& mask,
                  float scale = 1.0f) :
    input(input),
    mask(mask),
    scale(scale) {}

    const Buffer& input;
    const Buffer& mask;
    float scale;

};

}  // namespace fastertransformer

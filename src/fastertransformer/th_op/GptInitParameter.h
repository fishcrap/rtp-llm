#pragma once

#include "src/fastertransformer/utils/layernorm_types.h"
#include "src/fastertransformer/utils/activation_types.h"
#include "src/fastertransformer/utils/RopeConfig.h"

#include <vector>
#include <map>

namespace fastertransformer {

enum QuantMethod {
    None                = 0,
    WeightOnlyPerCol    = 1,
    GptQ                = 2,
    Awq                 = 3,
    SmoothQuant         = 4,
    OmniQuant           = 5,
    PerTensorQuant      = 6
};

enum TaskType {
    DENSE_EMBEDDING    = 0,
    ALL_EMBEDDING      = 1,
    SPARSE_EMBEDDING   = 2,
    COLBERT_EMBEDDING  = 3,
    LANGUAGE_MODEL     = 4,
    SEQ_CLASSIFICATION = 5,
    RERANKER           = 6,
    LINEAR_SOFTMAX     = 7
};

struct RoleSpecialTokens {
public:
    std::vector<int64_t> token_ids_;
    std::vector<int64_t> eos_token_ids_;
};

struct QuantAlgo {
public:
    QuantAlgo() = default;
    QuantAlgo(QuantMethod method, int bits, int group_size)
        : quant_method_(method)
        , weight_bits_(bits)
        , group_size_(group_size)
    {}
    bool isWeightOnlyPerCol() const {
        return quant_method_ == WeightOnlyPerCol;
    }
    bool isPerTensorQuant() const {
        return quant_method_ == PerTensorQuant;
    }
    bool isGptq() const {
        return quant_method_ == GptQ;
    }
    bool isAwq() const {
        return quant_method_ == Awq;
    }
    bool isSmoothQuant() const {
        return quant_method_ == SmoothQuant;
    }
    bool isOmniQuant() const {
        return quant_method_ == OmniQuant;
    }
    bool isQuant() const {
        return quant_method_ != None;
    }
    bool isGroupwise() const {
        return group_size_ > 0;
    }
    QuantMethod getQuantMethod() const {
        return quant_method_;
    }
    int64_t getGroupSize() const {
        return group_size_;
    }
    int64_t getWeightBits() const {
        return weight_bits_;
    }
    void setQuantAlgo(const std::string &method, int64_t bits, int64_t group_size);
private:
    QuantMethod quant_method_ = None;
    int64_t weight_bits_ = 0;
    int64_t group_size_ = 0;
};

struct SpecialTokens {
public:
    SpecialTokens();
    int64_t                               bos_token_id_ = -1;
    int64_t                               eos_token_id_ = 0;
    int64_t                               pad_token_id_ = 0;
    int64_t                               decoder_start_token_id_ = -1;
    RoleSpecialTokens user_;
    RoleSpecialTokens assistant_;
    RoleSpecialTokens system_;
    std::vector<std::vector<int64_t>>     stop_words_list_;
    std::vector<std::string>              stop_words_str_;
};

class GptInitParameter {
public:
    // model variant params used in ft
    int64_t head_num_               = 0;
    int64_t head_num_kv_            = -1;
    int64_t size_per_head_          = 0;
    int64_t inter_size_             = 0;
    int64_t inter_padding_size_     = -1;
    int64_t moe_inter_padding_size_ = -1;
    int64_t num_layers_             = 0;
    int64_t num_valid_layer_        = 0;
    int64_t hidden_size_            = 0;

    // in sparse, those params might vary among layers
    bool                 is_sparse_head_           = false;
    std::vector<int64_t> layer_head_num_           = {};
    std::vector<int64_t> layer_head_num_kv_        = {};
    std::vector<int64_t> layer_inter_size_         = {};
    std::vector<int64_t> layer_inter_padding_size_ = {};

    double             layernorm_eps_       = 1e-5;
    std::string        layernorm_type_str_  = "pre_layernorm";
    std::string        norm_type_str_       = "layernorm";
    std::string        activation_type_str_ = "Gelu";
    LayerNormType  layernorm_type_      = LayerNormType::pre_layernorm;
    NormType       norm_type_           = NormType::layernorm;
    TaskType       task_type_           = TaskType::LANGUAGE_MODEL;
    ActivationType activation_type_     = ActivationType::Gelu;

    int64_t rotary_embedding_dim_      = 0;
    int64_t rotary_embedding_style_    = 0;
    int64_t position_ids_style_        = 0;
    float rotary_embedding_base_     = 10000.f;
    double  rotary_embedding_scale_  = 1.0;
    double  rotary_factor1_          = 0;
    double  rotary_factor2_          = 0;
    int64_t org_embedding_max_pos_     = 0;
    // for Gemma, hidden_states = hidden_states * (hidden_size**0.5)
    double  input_embedding_scalar_    = 1;

    bool    use_logn_attn_ = false;
    double  q_scaling_ = 1;
    bool    qk_norm_ = false;

    bool    use_cross_attn_ = false;
    int64_t cross_attn_input_len_ = 0;

    bool use_norm_input_residual_    = false;
    bool use_norm_attn_out_residual_ = false;

    std::string data_type_         = "fp16";
    int64_t     local_rank_        = 0;

    int64_t              max_seq_len_                = 0;
    int64_t              vocab_size_                 = 0;
    int64_t              type_vocab_size_            = 0;
    int64_t              expert_num_                 = 0;
    int64_t              moe_k_                      = 0;
    bool                 moe_normalize_expert_scale_ = false;
    // 0 for no moe; 1 for all layer moe; 2 for partial layer moe
    int64_t              moe_style_                  = 0;
    std::vector<int64_t> moe_layer_index_            = {};

    bool has_positional_encoding_    = false;
    bool has_pre_decoder_layernorm_  = false;
    bool has_post_decoder_layernorm_ = false;
    bool has_lm_head_                = true;
    bool use_attention_linear_bias_  = false;
    bool use_fp32_to_compute_logit_  = false;
    bool add_bias_linear_            = false;
    bool has_moe_norm_               = false;
    double logit_scale_              = 1.0;
    bool is_causal_                  = true;
    bool use_kvcache_                = true;

    std::string tokenizer_path_    = "";
    std::string ckpt_path_         = "";
    int64_t     pre_seq_len_       = 0;
    bool        prefix_projection_ = false;
    bool        using_hf_sampling_ = false;

    SpecialTokens special_tokens_;
    QuantAlgo quant_algo_;

    // async mode config
    int64_t max_generate_batch_size_ = 1;
    int64_t max_context_batch_size_  = 1;
    int64_t gen_num_per_circle_      = 1;

    bool                 is_multimodal_ = false;
    std::vector<int64_t> mm_sep_tokens_ = {};
    bool                 include_sep_tokens_ = false;
    bool                 cal_mm_tokens_in_rotary_emb_ = true;

    bool     int8_kv_cache_       = false;
    bool     pre_allocate_op_mem_ = true;
    int64_t  seq_size_per_block_  = 8;

    int64_t  block_nums_                       = 0;
    int64_t  scheduler_reserve_resource_ratio_ = 5;
    int64_t  reserve_runtime_mem_mb_           = 0;
    int64_t  kv_cache_mem_mb_                  = 0;
    bool     reuse_cache_                      = false;
    bool     enable_partial_fallback_          = false;
    bool     enable_fast_gen_                  = false;
    int64_t  fast_gen_max_context_len_         = 0;

    bool use_medusa_ = false;
    bool use_expert_attention_ = false; // true for CogVLM2, false for other models

    std::string nccl_ip_        = "";
    int64_t     nccl_port_      = 0;
    int64_t     model_rpc_port_ = 0;
    int64_t     tp_size_        = 1;
    int64_t     tp_rank_        = 0;

    bool use_rpc_ = false;

    std::map<std::string, std::vector<int>> multi_task_prompt_tokens_;

    GptInitParameter();

    GptInitParameter(
        int64_t head_num, int64_t size_per_head, int64_t num_layers, int64_t max_seq_len,
        int64_t vocab_size, int64_t hidden_size);

    void insertMultiTaskPromptTokens(std::string task_id, std::vector<int64_t> tokens_id);
    void setLayerNormType();
    void setNormType();
    void setTaskType(std::string task);
    void setActivationType();
    bool isGatedActivation() const;

    RopeConfig getRopeConfig() const;
};

}

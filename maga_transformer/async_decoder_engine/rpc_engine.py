from typing import AsyncGenerator, Optional, Dict
import asyncio
import logging
from typing_extensions import override
from maga_transformer.models.propose_model.propose_model import ProposeModel
from maga_transformer.utils.time_util import Timer
from maga_transformer.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from maga_transformer.models.base_model import BaseModel, GenerateInput, GenerateOutputs
from maga_transformer.async_decoder_engine.base_engine import BaseEngine, KVCacheInfo
from maga_transformer.cpp.model_rpc.model_rpc_client import ModelRpcClient
from maga_transformer.utils.mm_process_engine import MMProcessEngine

class RPCEngine(BaseEngine):
    def __init__(self,
                 model: BaseModel,
                 propose_model: Optional[ProposeModel] = None) -> None:
        self.model = model
        self.propose_model = propose_model
        self.tokenizer = model.tokenizer
        self.config = model.config
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model)
        else:
            self.mm_engine = None
        self.rtp_llm_op_ = RtpLLMOp(model, self.mm_engine, propose_model)
        self.model_rpc_client = ModelRpcClient()

    @override
    def start(self) -> None:
        self.rtp_llm_op_.start()

    @override
    def stop(self) -> None:
        self.rtp_llm_op_.stop()

    @override
    def decode(self,
               input: GenerateInput) -> AsyncGenerator[GenerateOutputs, None]:
        return self.model_rpc_client.enqueue(input)

    @override
    def get_kv_cache_info(self) -> KVCacheInfo:
        available_kv_cache, total_kv_cache = self.rtp_llm_op_.get_kv_cache_info()
        return KVCacheInfo(available_kv_cache=available_kv_cache, total_kv_cache=total_kv_cache)

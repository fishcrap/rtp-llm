import gc
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from maga_transformer.config.exceptions import (ExceptionType,
                                                FtRuntimeException)
from maga_transformer.config.generate_config import RequestFormat
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters, VitParameters
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.base_model import EmbeddingOutput
from maga_transformer.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from maga_transformer.models.multimodal.multimodal_trt_engine import MultiModalTRTEngine
from maga_transformer.utils.database import CkptDatabase
from maga_transformer.utils.model_weight import ModelDeployWeightInfo, CkptWeightInfo, WeightInfo, sp_id, identity
from maga_transformer.utils.model_weights_loader import get_model_weights_loader

from maga_transformer.ops.comm.nccl_op import NcclOp
from maga_transformer.utils.util import to_cuda


class BaseVitWeights:
    def __init__(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        self.weight_names: List[str] = []
        self._set_weight_prefix()
        self._get_vit_params(vit_part, with_prefix)

    def _set_weight_prefix(self):
        self._ckpt_prefix = "model."
        self._ft_prefix = "self.mm_part."

    @property
    def ckpt_prefix(self) -> str:
        return self._ckpt_prefix

    @property
    def ft_prefix(self) -> str:
        return self._ft_prefix

    @ft_prefix.setter
    def ft_prefix(self, prefix: str) -> None:
        self._ft_prefix = prefix

    def _get_vit_params(self, vit_part: Dict[str, Any], with_prefix: bool = False):
        for vit_name, vit in vit_part.items():
            if isinstance(vit, torch.nn.Module):
                if len(vit_part) >= 2 or with_prefix:
                    self.weight_names.extend([vit_name + '.' + w for w in vit.state_dict().keys()])
                else:
                    self.weight_names.extend(list(vit.state_dict().keys()))
            elif isinstance(vit, torch.nn.Parameter):
                self.weight_names.append(vit_name)
            else:
                raise Exception("Unknown vit part type")


class BaseMultiModalWeightInfo:
    def __init__(self, config: GptInitModelParameters):
        self.vit_weights: Optional[BaseVitWeights] = config.vit_related_params.vit_weights

    def _get_vit_info(self, llm_weights: ModelDeployWeightInfo):
        # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
        # After supporting TP mm network, we will remove the check here.
        if self.vit_weights is not None and g_parallel_info.tp_rank == 0:
            weight_names = self.vit_weights.weight_names
            ckpt_prefix = self.vit_weights.ckpt_prefix

            for w in weight_names:
                w_name = ckpt_prefix + w
                llm_weights.weights.append(WeightInfo(w_name, [CkptWeightInfo(w_name, identity)], identity))
                llm_weights.tp_strategy[w_name] = sp_id


class MultiModalMixin:
    mm_part: MultiModalEmbeddingInterface
    nccl_op_: NcclOp

    @staticmethod
    def process_encode_plugin(prompt: str, generate_config: Dict[str, Any], tokenizer: Any, add_special_tokens: bool, **kwargs: Any) -> List[int]:
        if len(prompt) == 0:
            raise FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "prompt should have at least one token!")
        if type(prompt) is not str:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "expect string prompt, actual: " + str(prompt))
        if add_special_tokens:
            return tokenizer.encode(prompt)
        else:
            # for CogVLM2, we need to pass add_special_tokens=False to tokenizer
            return tokenizer.encode(prompt, add_special_tokens=False)

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], images: List[str],
                                        img_token: str, **kwargs: Any) -> Tuple[str, List[str]]:
        # should delete after chatapi interface update
        if kwargs.get('generate_config', {})['request_format'] == RequestFormat.CHAT_API:
            if isinstance(prompt, str):
                messages = json.loads(prompt, strict=False)
            else:
                messages = prompt
            new_prompt: str = ""
            new_images: List[str] = []
            for message in messages:
                new_prompt += message['role'].upper() + ' :'
                if isinstance(message['content'], str):
                    new_prompt += message['content'] + '\n'
                elif isinstance(message['content'], List):
                    for x in message['content']:
                        if x['type'] == 'text':
                            new_prompt += x['text']
                        elif x['type'] == 'image_url':
                            now_images = x['image_url']
                            if isinstance(now_images, List):
                                new_images.extend(now_images)
                                new_prompt += (img_token + '\n') * len(now_images)
                            else:
                                new_images.append(now_images)
                                new_prompt += img_token + '\n'
                        else:
                            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "content type can only be text or image_url, but get: " + x['type'])
                    new_prompt += '\n'
            return new_prompt + 'ASSISTANT :', new_images
        elif isinstance(prompt, List):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "raw request format cannot accept dict prompt")
        return prompt, images

    def expand_token_id(self, token_ids: List[int], images: List[torch.Tensor]) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        raise NotImplementedError()

    def _load_mm_weight(
        self, weights_info: ModelDeployWeightInfo, ckpt_path: str, vit_params: VitParameters,
        device: Union[str, torch.device], dtype: torch.dtype
    ):
        # Load weight only for self.mm_part

        database = CkptDatabase(ckpt_path)
        weight_loader = get_model_weights_loader(weights_info, database, compute_dtype=dtype)
        vit_weight = vit_params.vit_weights
        ckpt_prefix= vit_weight.ckpt_prefix
        ft_prefix = vit_weight.ft_prefix
        vit_weight_names = vit_weight.weight_names

        for vit_weight_name in vit_weight_names:
            ckpt_weight_name = ckpt_prefix + vit_weight_name
            param_name = ft_prefix + vit_weight_name
            param_name = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', param_name)
            tensor = weight_loader.load_tensor(ckpt_weight_name)[0]
            param = eval(param_name)
            param.data = tensor.reshape(param.data.shape).to(dtype).to(device)

    def init_mm_trt(
            self, weights_info: ModelDeployWeightInfo, ckpt_path: str,
            vit_params: VitParameters, device: Union[str, torch.device], dtype: torch.dtype
    ):
        # check whether VIT tensorrt exist
        try:
            import tensorrt
        except ImportError:
            raise RuntimeError("tensorrt library not fonnd")

        try:
            # TODO(xyz): currently model_name_path is ugly, we should let model_name_path passed by the frontend in
            # environment variable
            model_name_path = ckpt_path.replace('/', '_')

            visual_trt_engine = MultiModalTRTEngine(
                model_name_path, vit_params.config.get("image_size"), device, dtype
            )

            # TRT engine doesn't support TP, here we only generate trt engine on rank0 if trt engine is not cached
            if g_parallel_info.tp_rank == 0 and (
                (not MultiModalTRTEngine.trt_engine_cached(model_name_path, dtype))
                or os.environ.get("TRT_CACHE_ENABLED", "0") == "0"
            ):
                self._load_mm_weight(weights_info, ckpt_path, vit_params, device, dtype)

                # create cached dir if not exists
                output_dir = MultiModalTRTEngine.cache_path(model_name_path, dtype)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                visual_trt_engine.export_onnx(self.mm_part.vit, g_parallel_info.tp_size)

                # eagerly gc VIT network, release GPU memory for generating trt engine
                self.gc_mm_part(vit_params)

                visual_trt_engine.generate_trt_engine()

                # create a completion file to mark that the trt engine has been generated and cached
                MultiModalTRTEngine.completion_file_path(model_name_path, dtype).touch()

            # for TP > 1, only rank0 will generate trt engine, other ranks will wait rank0 to generate trt engine
            if g_parallel_info.tp_size > 1:
                self.nccl_op_.barrier(torch.device(device))

            self.gc_mm_part(vit_params)
            # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
            # After supporting TP mm network, we will remove the check here.
            if g_parallel_info.tp_rank == 0:
                visual_trt_engine.load_trt_engine()
                self.mm_part = visual_trt_engine

        except Exception as e:
            raise RuntimeError(f"init multimodal trt error: {e}")
        
    def gc_mm_part(self, vit_params: VitParameters):
        del self.mm_part
        self.mm_part = None
        vit_params.vit_weights = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_mm_weight(self, ctype: str, device: str):
        # wait rank0 finish loading weight, otherwise gang_server will die
        if g_parallel_info.tp_size > 1:
            self.nccl_op_.barrier(torch.device(device))
        # Currently, the multimodel network isn't split between devices. Only Rank 0 loads the weights.
        # After supporting TP mm network, we will remove the check here.
        if g_parallel_info.tp_rank >= 1:
            return

        # For trt engine, we don't need to load weight since its weight is inside trt engine.
        if isinstance(self.mm_part, MultiModalTRTEngine):
            return

        vit_weight = self.config.vit_related_params.vit_weights
        ckpt_prefix = vit_weight.ckpt_prefix
        ft_prefix = vit_weight.ft_prefix
        weight_names = vit_weight.weight_names

        def _safe_load_from_module(param: torch.nn.Parameter, fname: str, ctype: torch.dtype):
            param.data = self.weight.steal_pytorch_weight(fname).reshape(param.data.shape).to(ctype).to(device)

        for w in weight_names:
            w_name = ft_prefix + w
            w_name = re.sub(r'\.\d+\.', lambda x: '[' + x.group(0)[1:-1] + '].', w_name)
            param = eval(w_name)
            _safe_load_from_module(param, ckpt_prefix + w, ctype)

    def async_input_word_embedding(self, inputs: torch.Tensor, images: List[torch.Tensor], token_type_ids: torch.Tensor):
        inputs = inputs.reshape(1, -1)
        if g_parallel_info.tp_size <= 1:
            return EmbeddingOutput(self.multimodal_embedding(inputs, images, token_type_ids).squeeze(0), None)

        # for tp_size > 1
        check_num: int = 998244353
        check_num2: int = 1000000007
        shape_hints = torch.IntTensor([
            check_num,
            inputs.shape[1],
            check_num2
        ])
        shape_hints = to_cuda(shape_hints)
        self.nccl_op_.broadcast_tp([shape_hints])
        torch.cuda.current_stream().synchronize()
        assert shape_hints[0] == check_num and shape_hints[-1] == check_num2, 'check sum error'

        if g_parallel_info.tp_rank == 0:
            embedding_tensor = self.multimodal_embedding(inputs, images, token_type_ids).squeeze(0)
        else:
            input_length = shape_hints[1]
            embedding_tensor = torch.zeros((input_length, self.config.head_num * self.config.size_per_head), dtype=self.dtype, device=self.device)
        self.nccl_op_.broadcast_tp([embedding_tensor])
        torch.cuda.current_stream().synchronize()
        return EmbeddingOutput(embedding_tensor, None)

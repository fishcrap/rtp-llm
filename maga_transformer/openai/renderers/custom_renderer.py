from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import os
import torch
import asyncio
import logging
from dataclasses import dataclass, field
from PIL import Image
from concurrent.futures import Future

from transformers import PreTrainedTokenizerBase

from maga_transformer.models.base_model import GenerateOutput, BaseModel, GenerateInput, GenerateOutputs, AuxInfo
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.gpt_init_model_parameters import TemplateType
from maga_transformer.utils.mm_process_engine import MMProcessEngine
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, UsageInfo, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, \
    RoleEnum, RendererInfo
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.utils.word_util import get_stop_word_slices, truncate_response_with_stop_words

@dataclass
class StreamResponseObject:
    choices: List[ChatCompletionResponseStreamChoice] = field(default_factory=list)
    usage: Optional[UsageInfo] = None
    aux_info: Optional[AuxInfo] = None

@dataclass
class RendererParams:
    model_type: str
    max_seq_len: int
    eos_token_id: int
    stop_word_ids_list: List[List[int]]
    template_type: TemplateType = TemplateType.chat

@dataclass
class RenderedInputs:
    input_ids: List[int] = field(default_factory=list)
    input_images: List[str] = field(default_factory=list)
    rendered_prompt: str = field(default_factory=str)

class CustomChatRenderer():
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 renderer_params: RendererParams,
    ):
        self.tokenizer = tokenizer
        self.model_type = renderer_params.model_type
        self.max_seq_len = renderer_params.max_seq_len
        self.eos_token_id = renderer_params.eos_token_id
        self.stop_word_ids_list = renderer_params.stop_word_ids_list
        self.stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in self.stop_word_ids_list
        ]

        # NOTE: stop words or their ids only need to be added to one of these two lists.
        self.extra_stop_words: List[str] = []
        self.extra_stop_word_ids_list: List[List[int]] = []

    def __str__(self) -> str:
        return str(self.get_renderer_info())

    def __repr__(self) -> str:
        return self.__str__()

    def get_renderer_info(self) -> RendererInfo:
        extra_stop_word_ids_list = self.get_all_extra_stop_word_ids_list()
        extra_stop_words_list = [
            self.tokenizer.decode(stop_word_ids) for stop_word_ids in extra_stop_word_ids_list
        ]
        if len(extra_stop_words_list) and isinstance(extra_stop_words_list[0], list):
            extra_stop_words_list = [l[0] for l in extra_stop_words_list]
        return RendererInfo(
            class_name=self.__class__.__name__,
            renderer_model_type=self.model_type,
            extra_stop_word_ids_list=extra_stop_word_ids_list,
            extra_stop_words_list=extra_stop_words_list,
        )

    def add_extra_stop_words(self, extra_stop_words: List[str]):
        self.extra_stop_words.extend(extra_stop_words)

    def add_extra_stop_word_ids(self, extra_stop_word_ids: List[List[int]]):
        self.extra_stop_word_ids_list.extend(extra_stop_word_ids)

    def tokenize_words(self, words: List[str]) -> List[List[int]]:
        ids_list = []
        for word in words:
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                token_id = self.tokenizer.convert_tokens_to_ids(word)
                if isinstance(token_id, int):
                    ids_list.append([token_id])
                elif isinstance(token_id, list):
                    ids_list.append(token_id)
                else:
                    ids_list.append(self.tokenizer.encode(word, add_special_tokens=True))
            else:
                ids_list.append(self.tokenizer.encode(word))
        return ids_list

    def get_all_extra_stop_word_ids_list(self) -> List[List[int]]:
        ids_list_from_words = self.tokenize_words(self.extra_stop_words)
        return self.extra_stop_word_ids_list + ids_list_from_words

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        raise NotImplementedError

    async def generate_choice(
            self,
            request_id: int,
            input_ids: List[int],
            images: List[Future[Image.Image]],
            generate_config: GenerateConfig,
            model: Union[AsyncModel, BaseModel],
            request: ChatCompletionRequest
    ) -> AsyncGenerator[StreamResponseObject, None]:

        token_type_ids = []
        if model.is_multimodal() and len(images) > 0 and os.environ.get("USE_RPC_MODEL", "0") != "1":
            tasks = [asyncio.create_task(MMProcessEngine.get(images))]
            await asyncio.wait(tasks)
            images = tasks[0].result()

        if model.is_multimodal() and os.environ.get("USE_RPC_MODEL", "0") != "1":
            input_ids, images, token_type_ids = model.expand_token_id(input_ids, images)

        input_id_tensor = torch.Tensor(input_ids).int().unsqueeze(0)

        generate_config.is_streaming = True
        output_generator: AsyncGenerator[GenerateOutput, None] = model.enqueue(
            GenerateInput(
                request_id=request_id,
                token_ids=input_id_tensor,
                images=images,
                generate_config=generate_config,
                tokenizer=self.tokenizer,
                token_type_ids=token_type_ids
            )
        )

        async for response in self.render_response_stream(output_generator,
                                                          request,
                                                          generate_config):
            yield response

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutputs, None],
            request: ChatCompletionRequest,
            generate_config: GenerateConfig
    ) -> AsyncGenerator[StreamResponseObject, None]:
        index = 0
        output_token_length = 0
        responded_output_ids = []
        responded_string = ""
        finish_reason = None
        last_token_length = 0
        delta_output_string = ""
        stop_word_slice_list = get_stop_word_slices(generate_config.stop_words_str)

        def generate_stream_response(index: int, output_str: str, input_token_length: int, output_token_length: int):
            return StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            content=output_str,
                        ),
                    )],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length
                    )
                )

        output: GenerateOutput = None
        output_tokens_list = torch.empty(0, dtype=torch.int32)
        async for outputs in output_generator:
            if index == 0:
                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            role=RoleEnum.assistant,
                        ),
                    )]
                )

            index += 1
            output = outputs.generate_outputs[0]
            # all model incremental return output_ids
            input_token_length = output.aux_info.input_len
            output_tokens_list = torch.cat((output_tokens_list, output.output_ids), dim=1)
            output.output_ids = output_tokens_list
            output_ids = output.output_ids
            output_ids = self._clean_output_ids(output_ids)
            output_token_length = len(output_ids)
            finish_reason = self._check_finish_reason(output_ids, input_token_length, generate_config.max_new_tokens)
            output_ids = self._remove_stop_word_ids(output_ids)
            # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
            decoded_prev_token = self.tokenizer.decode(responded_output_ids[-last_token_length:])
            tokens_to_decode = responded_output_ids[-last_token_length:] + output_ids[len(responded_output_ids):]
            decoded_string = self.tokenizer.decode(tokens_to_decode)
            if (len(decoded_string)) and (u'\uFFFD' == decoded_string[-1]):
                continue
            delta_output_string = decoded_string[len(decoded_prev_token):]
            trunc_string = truncate_response_with_stop_words(delta_output_string, stop_word_slice_list)

            if len(delta_output_string) > 0 and trunc_string == delta_output_string:
                last_token_length = len(output_ids) - len(responded_output_ids)
                responded_output_ids = output_ids
                responded_string += delta_output_string
                stream_response = generate_stream_response(index, delta_output_string, input_token_length, output_token_length)
                delta_output_string = ""
                yield stream_response

        trunc_string = truncate_response_with_stop_words(delta_output_string, generate_config.stop_words_str)
        if len(delta_output_string) > 0 and trunc_string == delta_output_string:
            responded_string += delta_output_string
            yield generate_stream_response(index, delta_output_string, input_token_length, output_token_length)

        if finish_reason == None:
            logging.debug(f"output [{responded_string}] found no stop reason! use stop as default.")
            finish_reason = FinisheReason.stop

        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index + 1,
                delta=DeltaMessage(
                    content="",
                ),
                finish_reason=finish_reason
            )],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length
            ),
            aux_info=output.aux_info if request.aux_info else None
        )

    def _check_finish_reason(self, token_ids: List[int], input_token_length: int, max_new_tokens: int = -1) -> Optional[FinisheReason]:
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_word_ids_list
        if max_new_tokens > 0 and len(token_ids) >= max_new_tokens:
            return FinisheReason.length
        if len(token_ids) + input_token_length >= self.max_seq_len:
            return FinisheReason.length
        if token_ids and token_ids[-1] == self.eos_token_id:
            return FinisheReason.stop
        for stop_word_ids in stop_word_ids_list_all:
            if (len(token_ids) >= len(stop_word_ids)) and (token_ids[-len(stop_word_ids):] == stop_word_ids):
                return FinisheReason.stop
        return None

    def _remove_stop_word_ids(self, output_ids: List[int]) -> List[int]:
        stop_word_ids_list_all = self.get_all_extra_stop_word_ids_list() + self.stop_word_ids_list
        for stop_word_ids in stop_word_ids_list_all:
            #  此处应该从最大的范围开始判断
            # 有可能会有stopword_ids 重复的情况，比如[144575, 14098, 144575]
            # 若从1开始判断会导致 去除了最后一个 144575 就退出了
            for i in range(len(stop_word_ids) + 1, 1, -1):
                if output_ids[-i:] == stop_word_ids[:i]:
                    output_ids = output_ids[:-i]
                    break
        return output_ids

    def _clean_output_ids(self, output_ids_tensor: torch.Tensor) -> list[int]:
        output_ids_tensor = output_ids_tensor.cpu().reshape([-1])
        # TODO(wangyin): This slicing shouldn't be done here.
        # model should return output length, ids should be sliced with output length.
        output_ids = output_ids_tensor[output_ids_tensor != self.eos_token_id].tolist()
        return output_ids


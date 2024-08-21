from fastapi import Request
import torch
from typing import Union, Optional, List, Dict, Generator, Coroutine, AsyncGenerator, Any, Iterator
import os
import logging
from dataclasses import dataclass
from functools import partial

from transformers import PreTrainedTokenizerBase
from transformers.generation.stopping_criteria import StoppingCriteria

from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from transformers import PreTrainedTokenizerBase
from maga_transformer.models.base_model import BaseModel, GenerateOutput, GenerateResponse, GenerateInput
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.openai.api_datatype import ModelCard, ModelList, ChatMessage, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, UsageInfo, \
    FinisheReason, DeltaMessage, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, \
    DebugInfo
from maga_transformer.openai.renderers.custom_renderer import RendererParams, \
    StreamResponseObject, RenderedInputs, CustomChatRenderer
from maga_transformer.openai.renderer_factory import ChatRendererFactory
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.mm_process_engine import MMProcessEngine

class OpenaiEndopoint():
    def __init__(self, model: Union[AsyncModel, BaseModel]):
        self.model = model
        self.max_seq_len = self.model.config.max_seq_len
        if self.model.is_multimodal():
            if isinstance(self.model, AsyncModel):
                self.vit_engine = MMProcessEngine(self.model.model)
            else:
                self.vit_engine = MMProcessEngine(self.model)

        tokenizer = self.model.tokenizer
        if (tokenizer == None):
            raise AttributeError(f"model [{model}] has no tokenizer!")
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        self.eos_token_id = None
        if (isinstance(tokenizer, PreTrainedTokenizerBase)):
            self.eos_token_id = tokenizer.eos_token_id
        if (self.eos_token_id == None):
            self.eos_token_id = self.model.config.special_tokens.eos_token_id

        self.stop_word_ids_list = self.model.config.special_tokens.stop_words_list

        render_params = RendererParams(
            model_type=os.environ["MODEL_TYPE"],
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_word_ids_list,
            template_type=self.model.config.template_type
        )

        self.chat_renderer: CustomChatRenderer = ChatRendererFactory.get_renderer(self.tokenizer, render_params)
        logging.info(f"Finally openai endpoint uses renderer: {self.chat_renderer} ")
        self.template_renderer: CustomChatRenderer = self.chat_renderer \
            if (type(self.chat_renderer) == BasicRenderer) \
            else BasicRenderer(self.tokenizer, render_params)
        logging.info(f"chat_renderer [{self.chat_renderer}] is created.")
        extra_stop_word_ids_list = self.chat_renderer.get_all_extra_stop_word_ids_list()
        self.stop_word_ids_list.extend(extra_stop_word_ids_list)
        self.stop_words_list = []
        for stop_word_ids in self.stop_word_ids_list:
            word = self.tokenizer.decode(stop_word_ids)
            if len(word):
                self.stop_words_list.append(word)
        logging.info(f"use stop_words_list [{self.stop_words_list}]")

    async def list_models(self):
        global model_args
        model_card = ModelCard(id=self.model.__class__.__name__)
        return ModelList(data=[model_card])

    def _extract_generation_config(self, request: ChatCompletionRequest) -> GenerateConfig:
        # TODO(wangyin): implement this
        config = request.extra_configs or GenerateConfig()
        if request.stream != None:
            config.is_streaming = request.stream
        if request.temperature != None:
            config.temperature = request.temperature
        if request.top_p != None:
            config.top_p = request.top_p
        if request.max_tokens != None:
            config.max_new_tokens = request.max_tokens
        request_stop_words_list = request.stop if request.stop != None else []
        if isinstance(request_stop_words_list, str):
            request_stop_words_list = [request_stop_words_list]
        config.stop_words_str = self.stop_words_list + request_stop_words_list
        config.stop_words_list = self.stop_word_ids_list + self.chat_renderer.tokenize_words(request_stop_words_list)
        if request.chat_id != None:
            config.chat_id = request.chat_id
        if request.seed != None:
            config.random_seed = request.seed
        config.add_special_tokens(self.model.config.special_tokens)
        config.convert_select_tokens(self.model.config.vocab_size, self.tokenizer)
        return config

    async def _collect_complete_response(
            self,
            choice_generator: Optional[AsyncGenerator[StreamResponseObject, None]],
            debug_info: Optional[DebugInfo]) -> ChatCompletionResponse:
        all_choices = []
        usage = None
        aux_info = None
        async for response in choice_generator:
            if len(response.choices) != len(all_choices):
                if (all_choices == []):
                    all_choices = [
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(
                                role=choice.delta.role or RoleEnum.assistant,
                                content=choice.delta.content or None,
                                function_call=choice.delta.function_call or None,
                            ),
                            finish_reason=choice.finish_reason
                        ) for choice in response.choices
                    ]
                else:
                    raise ValueError(f"response.choices has different length! "
                                     f"[{response.choices}] vs [{all_choices}].")
            else:
                for i in range(len(all_choices)):
                    if all_choices[i].message.content == None:
                        all_choices[i].message.content = (response.choices[i].delta.content or None)
                    else:
                        all_choices[i].message.content += (response.choices[i].delta.content or "")
                    all_choices[i].message.role = response.choices[i].delta.role or all_choices[i].message.role
                    all_choices[i].message.function_call = response.choices[i].delta.function_call or all_choices[i].message.function_call
                    all_choices[i].finish_reason = response.choices[i].finish_reason or all_choices[i].finish_reason
            usage = response.usage or usage
            aux_info = response.aux_info or aux_info

        if (usage == None):
            logging.warn(f"No usage returned from stream response. use empty value.")
            usage = UsageInfo(
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens=0
            )
        return ChatCompletionResponse(
            choices=all_choices,
            usage=usage,
            aux_info=aux_info,
            model=self.model.__class__.__name__,
            debug_info=debug_info,
        )

    def _complete_stream_response(
            self, choice_generator: AsyncGenerator[StreamResponseObject, None],
            debug_info: Optional[DebugInfo]
    ) -> CompleteResponseAsyncGenerator:
        async def response_generator():
            debug_info_responded = False
            async for response in choice_generator:
                yield ChatCompletionStreamResponse(
                    choices=response.choices,
                    usage=response.usage,
                    aux_info=response.aux_info,
                    debug_info=debug_info if not debug_info_responded else None
                )
                debug_info_responded = True

        complete_response_collect_func = partial(self._collect_complete_response, debug_info=debug_info)
        return CompleteResponseAsyncGenerator(response_generator(), complete_response_collect_func)

    def _get_debug_info(self, renderer: CustomChatRenderer,
                        renderered_input: RenderedInputs, gen_config: GenerateConfig) -> DebugInfo:
        if renderered_input.rendered_prompt != "":
            prompt = renderered_input.rendered_prompt
        else:
            prompt = self.tokenizer.decode(renderered_input.input_ids)
        return DebugInfo(
            input_prompt=prompt,
            input_ids=renderered_input.input_ids,
            input_images=renderered_input.input_images,
            tokenizer_info=str(self.tokenizer),
            max_seq_len=self.max_seq_len,
            eos_token_id=self.eos_token_id,
            stop_word_ids_list=self.stop_word_ids_list,
            stop_words_list=self.stop_words_list,
            renderer_info=self.chat_renderer.get_renderer_info(),
            generate_config=gen_config
        )

    def chat_completion(
            self, request_id: int, chat_request: ChatCompletionRequest, raw_request: Request
    ) -> CompleteResponseAsyncGenerator:
        renderer = self.template_renderer if chat_request.user_template else self.chat_renderer
        rendered_input = renderer.render_chat(chat_request)
        input_ids = rendered_input.input_ids

        input_images = rendered_input.input_images
        generate_config = self._extract_generation_config(chat_request)

        if self.model.is_multimodal():
            if os.environ.get("USE_RPC_MODEL", "0") != "1":
                images = self.vit_engine.submit(input_images)
            else:
                images = input_images
        else:
            images = []

        debug_info = self._get_debug_info(renderer, rendered_input, generate_config) \
            if chat_request.debug_info else None
        choice_generator = renderer.generate_choice(
            request_id,
            input_ids,
            images,
            generate_config,
            self.model,
            chat_request
        )

        return self._complete_stream_response(choice_generator, debug_info)

    def chat_render(self, chat_request: ChatCompletionRequest) -> DebugInfo:
        renderer = self.template_renderer if chat_request.user_template else self.chat_renderer
        rendered_input = renderer.render_chat(chat_request)
        generate_config = self._extract_generation_config(chat_request)
        debug_info = self._get_debug_info(renderer, rendered_input, generate_config)
        return debug_info

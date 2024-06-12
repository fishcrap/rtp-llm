from .gpt import GPT
from .gpt_neox import GPTNeox
from .llama import Llama, Baichuan
from .sgpt_bloom import SGPTBloom
from .sgpt_bloom_vector import SGPTBloomVector
from .starcoder import StarCoder
from .starcoder2 import StarCoder2
from .bloom import Bloom
from .chat_glm_v2 import ChatGlmV2
from .chat_glm_v3 import ChatGlmV3
from .chat_glm_v4 import ChatGlmV4
from .qwen import QWen_7B, QWen_13B, QWen_1B8
from .qwen_v2 import QWenV2
from .falcon import Falcon
from .mpt import Mpt
from .phi import Phi
from .llava import Llava
from .qwen_vl import QWen_VL
from .mixtral import Mixtral
from .bert import Bert
from .megatron_bert import MegatronBert
from .qwen_v2_moe import Qwen2Moe

import logging
try:
    from internal_source.maga_transformer.models import internal_init
except ImportError:
    logging.info('no internal_source found')
    pass

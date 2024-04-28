import dataclasses
import datetime
import enum
from typing import Dict, List, Optional, Tuple, Type, Set, Union


class RunType(enum.Enum):
  PRODUCTION = 1
  TEST = 2


class CallType(str, enum.Enum):
  GENERATE_TEXT = 'generate_text'


class Provider(str, enum.Enum):
  OPENAI = 'openai'
  CLAUDE = 'claude'
  GEMINI = 'gemini'
  COHERE = 'cohere'
  DATABRICKS = 'databricks'
  MISTRAL = 'mistral'
  HUGGING_FACE = 'hugging_face'


class ProviderModel(str, enum.Enum):
  """Base provider model type."""


class OpenAIModel(ProviderModel):
  """OpenAI models.

  Models provided by OpenAI:
  https://platform.openai.com/docs/guides/text-generation
  """
  # Newer models (2023–)
  GPT_4 = 'gpt-4'
  GPT_4_TURBO_PREVIEW = 'gpt-4-turbo-preview'
  GPT_3_5_TURBO = 'gpt-3.5-turbo'

  # Updated legacy models (2023)
  GPT_3_5_TURBO_INSTRUCT = 'gpt-3.5-turbo-instruct'
  BABBAGE = 'babbage-002'
  DAVINCI = 'davinci-002'


class ClaudeModel(ProviderModel):
  """Claude models.

  Models provided by Claude:
  https://claude.ai/docs/models
  """
  # Latest models (03/28/2024)
  CLAUDE_3_OPUS =  'claude-3-opus-20240229'
  CLAUDE_3_SONNET = 'claude-3-sonnet-20240229'
  CLAUDE_3_HAIKU = 'claude-3-haiku-20240307'


class GeminiModel(ProviderModel):
  """Gemini models.

  Models provided by Gemini:
  https://ai.google.dev/models/gemini
  """
  GEMINI_1_0_PRO = 'models/gemini-1.0-pro'
  GEMINI_1_0_PRO_001 =  'models/gemini-1.0-pro-001'
  GEMINI_1_0_PRO_LATEST = 'models/gemini-1.0-pro-latest'
  GEMINI_1_0_PRO_VISION_LATEST = 'models/gemini-1.0-pro-vision-latest'
  GEMINI_1_5_PRO_LATEST = 'models/gemini-1.5-pro-latest'
  GEMINI_PRO = 'models/gemini-pro'
  GEMINI_PRO_VISION = 'models/gemini-pro-vision'


class CohereModel(ProviderModel):
  """Cohere models.

  Models provided by Cohere:
  https://docs.cohere.com/docs/models
  """
  COMMAND_LIGHT = 'command-light'
  COMMAND_LIGHT_NIGHTLY = 'command-light-nightly'
  COMMAND = 'command'
  COMMAND_NIGHTLY = 'command-nightly'
  COMMAND_R = 'command-r'


class DatabricksModel(ProviderModel):
  """Databricks models.

  Models provided by Databricks:
  https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis
  """
  DATABRICKS_DBRX_INSTRUCT = 'databricks-dbrx-instruct'
  DATABRICKS_MIXTRAL_8x7b_INSTRUCT = 'databricks-mixtral-8x7b-instruct'
  DATABRICKS_LLAMA_2_70b_CHAT = 'databricks-llama-2-70b-chat'


class MistralModel(ProviderModel):
  """Mistral models.

  Models provided by Mistral:
  https://docs.mistral.ai/platform/endpoints/
  """
  OPEN_MISTRAL_7B = 'open-mistral-7b'
  OPEN_MIXTRAL_8X7B = 'open-mixtral-8x7b'
  MISTRAL_SMALL_LATEST = 'mistral-small-latest'
  MISTRAL_MEDIUM_LATEST = 'mistral-medium-latest'
  MISTRAL_LARGE_LATEST = 'mistral-large-latest'


class HuggingFaceModel(ProviderModel):
  """Hugging Face models.

  Models provided by Hugging Face on HuggingFaceChat:
  https://huggingface.co/chat/models
  """
  # To be able to use Google models, you need to sign terms of service:
  GOOGLE_GEMMA_7B_IT = 'google/gemma-7b-it'
  # Requires pro subscription:
  # META_LLAMA_2_70B_CHAT_HF = 'meta-llama/Llama-2-70b-chat-hf'
  # Requires pro subscription:
  # CODELLAMA_70B_INSTRUCT_HF = 'codellama/CodeLlama-70b-Instruct-hf'
  MISTRAL_MIXTRAL_8X7B_INSTRUCT = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
  MISTRAL_MISTRAL_7B_INSTRUCT = 'mistralai/Mistral-7B-Instruct-v0.2'
  NOUS_HERMES_2_MIXTRAL_8X7B = 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
  OPENCHAT_3_5 = 'openchat/openchat-3.5-0106'


PROVIDER_MODEL_MAP: Dict[Provider, Type[ProviderModel]] = {
    Provider.OPENAI: OpenAIModel,
    Provider.CLAUDE: ClaudeModel,
    Provider.GEMINI: GeminiModel,
    Provider.COHERE: CohereModel,
    Provider.DATABRICKS: DatabricksModel,
    Provider.MISTRAL: MistralModel,
    Provider.HUGGING_FACE: HuggingFaceModel,
}

PROVIDER_KEY_MAP: Dict[Provider, List[str]] = {
    Provider.OPENAI: ['OPENAI_API_KEY'],
    Provider.CLAUDE: ['ANTHROPIC_API_KEY'],
    Provider.GEMINI: ['GOOGLE_API_KEY'],
    Provider.COHERE: ['CO_API_KEY'],
    Provider.DATABRICKS: ['DATABRICKS_TOKEN', 'DATABRICKS_HOST'],
    Provider.MISTRAL: ['MISTRAL_API_KEY'],
    Provider.HUGGING_FACE: ['HUGGINGFACE_API_KEY'],
}


GENERATE_TEXT_MODELS: Dict[Provider, List[Type[ProviderModel]]] = {
    Provider.OPENAI: [
        OpenAIModel.GPT_4,
        OpenAIModel.GPT_4_TURBO_PREVIEW,
        OpenAIModel.GPT_3_5_TURBO,
    ],
    Provider.CLAUDE: [
        ClaudeModel.CLAUDE_3_OPUS,
        ClaudeModel.CLAUDE_3_SONNET,
        ClaudeModel.CLAUDE_3_HAIKU,
    ],
    Provider.GEMINI: [
        GeminiModel.GEMINI_1_0_PRO,
        GeminiModel.GEMINI_1_0_PRO_001,
        GeminiModel.GEMINI_1_0_PRO_LATEST,
        GeminiModel.GEMINI_1_5_PRO_LATEST,
        GeminiModel.GEMINI_PRO,
    ],
    Provider.COHERE: [
        CohereModel.COMMAND_LIGHT,
        CohereModel.COMMAND_LIGHT_NIGHTLY,
        CohereModel.COMMAND,
        CohereModel.COMMAND_NIGHTLY,
        CohereModel.COMMAND_R,
    ],
    Provider.DATABRICKS: [
        DatabricksModel.DATABRICKS_DBRX_INSTRUCT,
        DatabricksModel.DATABRICKS_MIXTRAL_8x7b_INSTRUCT,
        DatabricksModel.DATABRICKS_LLAMA_2_70b_CHAT,
    ],
    Provider.MISTRAL: [
        MistralModel.OPEN_MISTRAL_7B,
        MistralModel.OPEN_MIXTRAL_8X7B,
        MistralModel.MISTRAL_SMALL_LATEST,
        MistralModel.MISTRAL_MEDIUM_LATEST,
        MistralModel.MISTRAL_LARGE_LATEST,
    ],
    Provider.HUGGING_FACE: [
        HuggingFaceModel.GOOGLE_GEMMA_7B_IT,
        HuggingFaceModel.MISTRAL_MIXTRAL_8X7B_INSTRUCT,
        HuggingFaceModel.MISTRAL_MISTRAL_7B_INSTRUCT,
        HuggingFaceModel.NOUS_HERMES_2_MIXTRAL_8X7B,
        HuggingFaceModel.OPENCHAT_3_5,
    ],
}


ModelType = Tuple[Provider, ProviderModel]
StopType = Union[str, List[str]]
MessagesType = List[Dict[str, str]]


@dataclasses.dataclass
class LoggingOptions:
  path: Optional[str] = None
  time: bool = True
  prompt: bool = True
  response: bool = True
  error: bool = True


class LoggingType(str, enum.Enum):
  QUERY = 'query'
  ERROR = 'error'
  WARNING = 'warning'
  INFO = 'info'


@dataclasses.dataclass
class CacheOptions:
  path: Optional[str] = None
  unique_response_limit: Optional[int] = 1
  duration: Optional[int] = None
  retry_if_error_cached: bool = False


@dataclasses.dataclass
class SummaryOptions:
  json: bool = True


@dataclasses.dataclass
class QueryRecord:
  call_type: Optional[CallType] = None
  model: Optional[ModelType] = None
  prompt: Optional[str] = None
  system: Optional[str] = None
  messages: Optional[MessagesType] = None
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  stop: Optional[StopType] = None
  hash_value: Optional[str] = None


@dataclasses.dataclass
class QueryResponseRecord:
  response: Optional[str] = None
  error: Optional[str] = None
  error_traceback: Optional[str] = None
  start_time: Optional[datetime.datetime] = None
  end_time: Optional[datetime.datetime] = None
  response_time: Optional[datetime.timedelta] = None


@dataclasses.dataclass
class CacheRecord:
  query_record: Optional[QueryRecord] = None
  query_responses: List[QueryResponseRecord] = dataclasses.field(
      default_factory=list)
  shard_id: Optional[str] = None
  last_access_time: Optional[datetime.datetime] = None
  call_count: Optional[int] = None


@dataclasses.dataclass
class LightCacheRecord:
  query_record_hash: Optional[str] = None
  query_response_count: Optional[int] = None
  shard_id: Optional[int] = None
  last_access_time: Optional[datetime.datetime] = None
  call_count: Optional[int] = None


class CacheLookFailReason(str, enum.Enum):
  CACHE_NOT_FOUND = 'cache_not_found'
  CACHE_NOT_MATCHED = 'cache_not_matched'
  UNIQUE_RESPONSE_LIMIT_NOT_REACHED = 'unique_response_limit_not_reached'
  PROVIDER_ERROR_CACHED = 'provider_error_cached'


@dataclasses.dataclass
class CacheLookResult:
  query_response: Optional[QueryResponseRecord] = None
  look_fail_reason: Optional[CacheLookFailReason] = None


class ResponseSource(str, enum.Enum):
  CACHE = 'cache'
  PROVIDER = 'provider'


@dataclasses.dataclass
class LoggingRecord:
  query_record: Optional[QueryRecord] = None
  response_record: Optional[QueryResponseRecord] = None
  response_source: Optional[ResponseSource] = None
  look_fail_reason: Optional[CacheLookFailReason] = None


@dataclasses.dataclass
class ModelStatus:
  unprocessed_models: Set[ModelType] = dataclasses.field(default_factory=set)
  working_models: Set[ModelType] = dataclasses.field(default_factory=set)
  failed_models: Set[ModelType] = dataclasses.field(default_factory=set)
  filtered_models: Set[ModelType] = dataclasses.field(default_factory=set)
  provider_queries: List[LoggingRecord] = (
      dataclasses.field(default_factory=list))

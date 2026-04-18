import dataclasses
import datetime
import enum
from typing import Any, Dict, List

import pydantic

import proxai.chat.message_content as message_content
import proxai.chat.message as message
import proxai.chat.chat_session as chat_session

ContentType = message_content.ContentType
MessageRoleType = message_content.MessageRoleType
PydanticContent = message_content.PydanticContent
MessageContent = message_content.MessageContent
Message = message.Message
Chat = chat_session.Chat
SUPPORTED_MEDIA_TYPES = message_content.SUPPORTED_MEDIA_TYPES


class RunType(enum.Enum):
  """Execution mode for the ProxAI client."""

  PRODUCTION = "PRODUCTION"
  TEST = "TEST"


class OutputFormatType(str, enum.Enum):
  """Output formats for capability filtering and model registration."""

  TEXT = "TEXT"
  IMAGE = "IMAGE"
  AUDIO = "AUDIO"
  VIDEO = "VIDEO"
  JSON = "JSON"
  PYDANTIC = "PYDANTIC"
  MULTI_MODAL = "MULTI_MODAL"


class InputFormatType(str, enum.Enum):
  """Input formats for capability filtering."""

  TEXT = "TEXT"
  IMAGE = "IMAGE"
  DOCUMENT = "DOCUMENT"
  AUDIO = "AUDIO"
  VIDEO = "VIDEO"
  JSON = "JSON"
  PYDANTIC = "PYDANTIC"


OutputFormatTypeParam = OutputFormatType | str
InputFormatTypeParam = InputFormatType | str


ProviderNameType = str
ModelNameType = str
RawProviderModelIdentifierType = str


@dataclasses.dataclass(frozen=True)
class ProviderModelType:
  """Immutable identifier for a specific provider and model combination.

  This type is returned by model discovery functions like
  px.models.list_models() and px.models.get_model(). It uniquely identifies
  a model by its provider,
  model name, and the provider's internal identifier.

  Attributes:
    provider: The AI provider name (e.g., 'openai', 'anthropic', 'google').
    model: The model name (e.g., 'gpt-4', 'claude-3-opus', 'gemini-pro').
    provider_model_identifier: The provider's internal model identifier,
      which may differ from the model name.

  Example:
    >>> import proxai as px
    >>> models = px.models.list_models()
    >>> for model in models:
    ...   print(f"{model.provider}: {model.model}")
    openai: gpt-4
    anthropic: claude-3-opus
    >>> # Type checking
    >>> model = px.models.get_model("openai", "gpt-4")
    >>> isinstance(model, px.ProviderModelType)
    True
  """

  provider: ProviderNameType
  model: ModelNameType
  provider_model_identifier: RawProviderModelIdentifierType

  def __str__(self):  # noqa: D105
    return f"({self.provider}, {self.model})"

  def __repr__(self):  # noqa: D105
    return (
        "ProviderModelType("
        f"provider={self.provider}, "
        f"model={self.model}, "
        f"provider_model_identifier={self.provider_model_identifier})"
    )

  def __lt__(self, other):  # noqa: D105
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) < str(other)

  def __gt__(self, other):  # noqa: D105
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) > str(other)

  def __le__(self, other):  # noqa: D105
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) <= str(other)

  def __ge__(self, other):  # noqa: D105
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) >= str(other)


ProviderModelTupleType = tuple[ProviderNameType, ModelNameType]
ProviderModelIdentifierType = ProviderModelType | ProviderModelTupleType
StopType = str | list[str]


@dataclasses.dataclass
class ProviderModelPricingType:
  """Cost information for a model's token usage."""

  input_token_cost: float | None = None
  output_token_cost: float | None = None


class FeatureSupportType(str, enum.Enum):
  """Support level for a feature."""

  SUPPORTED = "SUPPORTED"
  BEST_EFFORT = "BEST_EFFORT"
  NOT_SUPPORTED = "NOT_SUPPORTED"


@dataclasses.dataclass
class ParameterConfigType:
  """Parameter configuration for a provider endpoint."""

  temperature: FeatureSupportType | None = None
  max_tokens: FeatureSupportType | None = None
  stop: FeatureSupportType | None = None
  n: FeatureSupportType | None = None
  thinking: FeatureSupportType | None = None


@dataclasses.dataclass
class ToolConfigType:
  """Tool configuration for a provider endpoint."""

  web_search: FeatureSupportType | None = None


@dataclasses.dataclass
class OutputFormatConfigType:
  """Output format configuration for a provider endpoint."""

  text: FeatureSupportType | None = None
  image: FeatureSupportType | None = None
  audio: FeatureSupportType | None = None
  video: FeatureSupportType | None = None
  json: FeatureSupportType | None = None
  pydantic: FeatureSupportType | None = None
  multi_modal: FeatureSupportType | None = None


@dataclasses.dataclass
class InputFormatConfigType:
  """Input format configuration for a provider endpoint."""

  text: FeatureSupportType | None = None
  image: FeatureSupportType | None = None
  document: FeatureSupportType | None = None
  audio: FeatureSupportType | None = None
  video: FeatureSupportType | None = None
  json: FeatureSupportType | None = None
  pydantic: FeatureSupportType | None = None


@dataclasses.dataclass
class FeatureConfigType:
  """Feature configuration for a provider endpoint."""

  prompt: FeatureSupportType | None = None
  messages: FeatureSupportType | None = None
  system_prompt: FeatureSupportType | None = None
  add_system_to_messages: bool | None = None
  parameters: ParameterConfigType | None = None
  tools: ToolConfigType | None = None
  output_format: OutputFormatConfigType | None = None
  input_format: InputFormatConfigType | None = None


class ToolTag(str, enum.Enum):
  """What tools the model supports."""

  WEB_SEARCH = "web_search"


class FeatureTag(str, enum.Enum):
  """General model features and parameters."""

  PROMPT = "prompt"
  MESSAGES = "messages"
  SYSTEM_PROMPT = "system_prompt"
  TEMPERATURE = "temperature"
  MAX_TOKENS = "max_tokens"
  STOP = "stop"
  N = "n"
  THINKING = "thinking"


InputFormatTypeParam = (
    list[InputFormatType] | list[str] | InputFormatType | str | None)
OutputFormatTypeParam = (
    list[OutputFormatType] | list[str] | OutputFormatType | str | None)
ToolTagParam = list[ToolTag] | list[str] | ToolTag | str | None
FeatureTagParam = list[FeatureTag] | list[str] | FeatureTag | str | None


class ModelSizeType(str, enum.Enum):
  """Size category for AI models."""

  SMALL = "small"
  MEDIUM = "medium"
  LARGE = "large"
  LARGEST = "largest"


ModelSizeIdentifierType = ModelSizeType | str


@dataclasses.dataclass
class ProviderModelMetadataType:
  """Metadata describing a model's characteristics and capabilities."""

  is_recommended: bool | None = None
  model_size_tags: list[ModelSizeType] | None = None
  tags: list[str] | None = None


class ConfigOriginType(enum.Enum):
  """Source of the model configuration."""

  BUILT_IN = "BUILT_IN"
  PROXDASH = "PROXDASH"


# (provider, model) without model_signature
ProviderModelTupleParam = tuple[ProviderNameType, ModelNameType]
ProviderModelParam = ProviderModelTupleParam | ProviderModelType
MessagesParam = Dict[str, Any] | List[Dict[str, Any]] | chat_session.Chat


@dataclasses.dataclass
class ModelConfigsSchemaMetadataType:
  """Version and release information for model configurations."""

  version: str | None = None
  released_at: datetime.datetime | None = None
  min_proxai_version: str | None = None
  config_origin: ConfigOriginType | None = None
  release_notes: str | None = None


@dataclasses.dataclass
class ProviderModelConfig:
  provider_model: ProviderModelType
  pricing: ProviderModelPricingType
  features: FeatureConfigType
  metadata: ProviderModelMetadataType


OutputFormatTypeMappingType = dict[OutputFormatType, list[ProviderModelType]]
ModelSizeMappingType = dict[ModelSizeType, list[ProviderModelType]]
RecommendedModelsMappingType = dict[ProviderNameType, list[ProviderModelType]]
ProviderModelConfigsMappingType = dict[
    ProviderNameType, dict[ModelNameType, ProviderModelConfig]]


@dataclasses.dataclass
class ModelRegistry:
  metadata: ModelConfigsSchemaMetadataType
  default_model_priority_list: list[ProviderModelType]
  provider_model_configs: ProviderModelConfigsMappingType


@dataclasses.dataclass
class LoggingOptions:
  """Configuration for logging behavior.

  Args:
    logging_path: Directory path where log files will be written.
      If None, file logging is disabled.
    stdout: If True, logs are also printed to standard output.
      Defaults to False.
    hide_sensitive_content: If True, sensitive information like prompts
      and responses are redacted in logs. Defaults to False.

  Example:
    >>> import proxai as px
    >>> logging_opts = px.LoggingOptions(logging_path="/tmp/logs", stdout=True)
  """

  logging_path: str | None = None
  stdout: bool = False
  hide_sensitive_content: bool = False


class LoggingType(str, enum.Enum):
  """Severity level for log messages."""

  QUERY = "QUERY"
  ERROR = "ERROR"
  WARNING = "WARNING"
  INFO = "INFO"


@dataclasses.dataclass
class CacheOptions:
  """Configuration for query and model caching behavior.

  Controls how ProxAI caches query responses and model availability data
  to reduce API calls and improve performance.

  Args:
    cache_path: Directory path where cache files will be stored.
      If None, caching is disabled.
    unique_response_limit: Number of unique responses to collect for
      the same query before returning from cache. Useful for getting
      diverse outputs. Defaults to 1.
    retry_if_error_cached: If True, retries the API call when the
      cached response was an error. Defaults to False.
    clear_query_cache_on_connect: If True, clears the query cache
      when connect() is called. Defaults to False.
    disable_model_cache: If True, disables caching of model availability
      data. Defaults to False.
    clear_model_cache_on_connect: If True, clears the model cache
      when connect() is called. Defaults to False.
    model_cache_duration: Duration in seconds for which model cache
      entries remain valid. If None, uses the default duration.

  Example:
    >>> import proxai as px
    >>> cache_opts = px.CacheOptions(
    ...   cache_path="/tmp/proxai_cache", unique_response_limit=3
    ... )
  """

  cache_path: str | None = None

  unique_response_limit: int | None = 1
  retry_if_error_cached: bool = False
  clear_query_cache_on_connect: bool = False

  disable_model_cache: bool = False
  clear_model_cache_on_connect: bool = False
  model_cache_duration: int | None = None


@dataclasses.dataclass
class ProxDashOptions:
  """Configuration for ProxDash monitoring integration.

  ProxDash is a monitoring platform that tracks API usage, costs,
  and performance metrics for your ProxAI queries.

  Args:
    stdout: If True, prints ProxDash status messages to standard
      output. Defaults to False.
    hide_sensitive_content: If True, sensitive information like prompts
      and responses are not sent to ProxDash. Defaults to False.
    disable_proxdash: If True, completely disables ProxDash integration
      even if an API key is configured. Defaults to False.
    api_key: Your ProxDash API key. If None, looks for the
      PROXDASH_API_KEY environment variable.
    base_url: ProxDash server URL. Defaults to the production server.

  Example:
    >>> import proxai as px
    >>> proxdash_opts = px.ProxDashOptions(api_key="your-api-key", stdout=True)
  """

  stdout: bool = False
  hide_sensitive_content: bool = False
  disable_proxdash: bool = False
  api_key: str | None = None
  base_url: str | None = "https://proxainest-production.up.railway.app"


@dataclasses.dataclass
class SummaryOptions:
  """Output format options for summary reports."""

  json: bool = True


class FeatureMappingStrategy(str, enum.Enum):
  """Strategy for handling feature compatibility between requests and models.

  When a request includes features that a model doesn't fully support,
  this strategy determines how ProxAI should handle the incompatibility.

  Attributes:
    BEST_EFFORT: Attempts to map features even if not fully supported.
      For example, if a model doesn't support system messages natively,
      ProxAI may prepend it to the user prompt. This is the default.
    STRICT: Requires exact feature support. Raises an error if the
      requested features are not fully supported by the model.

  Example:
    >>> import proxai as px
    >>> # Use strict mode to ensure full feature support
    >>> px.connect(feature_mapping_strategy=px.FeatureMappingStrategy.STRICT)
    >>> # Or per-request
    >>> px.generate_text(
    ...   prompt="Hello",
    ...   feature_mapping_strategy=px.FeatureMappingStrategy.STRICT,
    ... )
  """

  BEST_EFFORT = "BEST_EFFORT"
  STRICT = "STRICT"


@dataclasses.dataclass
class RunOptions:
  """Combined runtime configuration for a ProxAI session."""

  run_type: RunType | None = None
  hidden_run_key: str | None = None
  experiment_path: str | None = None
  root_logging_path: str | None = None
  default_model_cache_path: str | None = None
  logging_options: LoggingOptions | None = None
  cache_options: CacheOptions | None = None
  proxdash_options: ProxDashOptions | None = None
  allow_multiprocessing: bool | None = None
  model_test_timeout: int | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None
  suppress_provider_errors: bool | None = None
  keep_raw_provider_response: bool | None = None


@dataclasses.dataclass
class OutputFormat:
  """Specification for the desired output format."""

  type: OutputFormatType | None = None
  pydantic_class: type[pydantic.BaseModel] | None = None
  pydantic_class_name: str | None = None
  pydantic_class_json_schema: dict | None = None


OutputFormatParam = str | type[pydantic.BaseModel] | OutputFormat


@dataclasses.dataclass
class ResultMediaContentType:
  """Media content returned by a provider in a result.

  Attributes:
    data: Raw media bytes.
    media_type: MIME type (e.g., "image/png", "audio/wav").
  """

  data: bytes
  media_type: str


class ThinkingType(str, enum.Enum):
  """Type of thinking for a query to a provider."""

  LOW = "LOW"
  MEDIUM = "MEDIUM"
  HIGH = "HIGH"


@dataclasses.dataclass
class ParameterType:
  """Parameters for a query to a provider."""

  temperature: float | None = None
  max_tokens: int | None = None
  stop: StopType | None = None
  n: int | None = None
  thinking: ThinkingType | None = None


class Tools(enum.Enum):
  """Tools for a query to a provider."""

  WEB_SEARCH = "WEB_SEARCH"


@dataclasses.dataclass
class ConnectionOptions:
  """Connection options for a query to a provider."""

  fallback_models: list[ProviderModelType] | None = None
  suppress_provider_errors: bool | None = None
  endpoint: str | None = None
  skip_cache: bool | None = None
  override_cache_value: bool | None = None


@dataclasses.dataclass
class QueryRecord:
  """Complete record of a query sent to a provider."""

  prompt: str | None = None
  chat: Chat | None = None
  system_prompt: str | None = None
  provider_model: ProviderModelType | None = None
  parameters: ParameterType | None = None
  tools: list[Tools] | None = None
  output_format: OutputFormat | None = None
  connection_options: ConnectionOptions | None = None
  hash_value: str | None = None


class ResultStatusType(str, enum.Enum):
  """Status of a query to a provider."""

  SUCCESS = "SUCCESS"
  FAILED = "FAILED"


@dataclasses.dataclass
class ChoiceType:
  """Choice of a query to a provider."""

  output_text: str | None = None
  output_image: message_content.MessageContent | None = None
  output_audio: message_content.MessageContent | None = None
  output_video: message_content.MessageContent | None = None
  output_json: dict | None = None
  output_pydantic: pydantic.BaseModel | None = None

  content: list[MessageContent] | None = None


@dataclasses.dataclass
class UsageType:
  """Usage of a query to a provider."""

  input_tokens: int | None = None
  output_tokens: int | None = None
  total_tokens: int | None = None
  estimated_cost: int | None = None


@dataclasses.dataclass
class ToolUsageType:
  """Usage of a tool for a query to a provider."""

  web_search_count: int | None = None
  web_search_citations: list[str] | None = None


@dataclasses.dataclass
class TimeStampType:
  """Timestamp information for a query to a provider."""

  start_utc_date: datetime.datetime | None = None
  end_utc_date: datetime.datetime | None = None
  local_time_offset_minute: int | None = None
  response_time: datetime.timedelta | None = None
  cache_response_time: datetime.timedelta | None = None


@dataclasses.dataclass
class ResultRecord:
  """Result of a query to a provider."""

  status: ResultStatusType | None = None

  role: MessageRoleType | None = None

  output_text: str | None = None
  output_image: message_content.MessageContent | None = None
  output_audio: message_content.MessageContent | None = None
  output_video: message_content.MessageContent | None = None
  output_json: dict | None = None
  output_pydantic: pydantic.BaseModel | None = None

  content: list[MessageContent] | None = None
  choices: list[ChoiceType] | None = None

  error: str | None = None
  error_traceback: str | None = None

  usage: UsageType | None = None
  tool_usage: ToolUsageType | None = None
  timestamp: TimeStampType | None = None


@dataclasses.dataclass
class ExecutorResult:
  """Return type for provider endpoint executors."""

  result_record: ResultRecord
  raw_provider_response: Any | None = None


class ResultSource(str, enum.Enum):
  """Origin of the response data."""

  CACHE = "CACHE"
  PROVIDER = "PROVIDER"


class CacheLookFailReason(str, enum.Enum):
  """Reason why a cache lookup did not return a result."""

  CACHE_NOT_FOUND = "CACHE_NOT_FOUND"
  CACHE_NOT_MATCHED = "CACHE_NOT_MATCHED"
  UNIQUE_RESPONSE_LIMIT_NOT_REACHED = "UNIQUE_RESPONSE_LIMIT_NOT_REACHED"
  PROVIDER_ERROR_CACHED = "PROVIDER_ERROR_CACHED"


@dataclasses.dataclass
class ConnectionMetadata:
  """Metadata for a cached query."""

  result_source: ResultSource | None = None
  cache_look_fail_reason: CacheLookFailReason | None = None
  endpoint_used: str | None = None
  failed_fallback_models: list[ProviderModelType] | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None


@dataclasses.dataclass
class DebugInfo:
  """Debug-only sidecar fields attached to a CallRecord.

  None of the fields here are part of ProxAI's stable contract. They exist
  as escape hatches for ad-hoc debugging and may change shape or disappear
  without notice. They are intentionally never serialized to the query
  cache or to ProxDash.
  """

  raw_provider_response: Any | None = None


@dataclasses.dataclass
class CallRecord:
  """Complete record of a call to a provider."""

  query: QueryRecord | None = None
  result: ResultRecord | None = None
  connection: ConnectionMetadata | None = None
  debug: DebugInfo | None = None


@dataclasses.dataclass
class CacheRecord:
  """Cached query and its associated responses."""

  query: QueryRecord | None = None
  results: list[ResultRecord] = dataclasses.field(default_factory=list)
  shard_id: str | None = None
  last_access_time: datetime.datetime | None = None
  call_count: int | None = None


@dataclasses.dataclass
class LightCacheRecord:
  """Lightweight cache metadata without full response data."""

  query_hash: str | None = None
  results_count: int | None = None
  shard_id: int | None = None
  last_access_time: datetime.datetime | None = None
  call_count: int | None = None


@dataclasses.dataclass
class CacheLookResult:
  """Result of a cache lookup operation."""

  result: ResultRecord | None = None
  cache_look_fail_reason: CacheLookFailReason | None = None


@dataclasses.dataclass
class ModelStatus:
  """Tracking status of models during availability testing."""

  unprocessed_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set
  )
  working_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set
  )
  failed_models: set[ProviderModelType] = dataclasses.field(default_factory=set)
  filtered_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set
  )
  provider_queries: dict[ProviderModelType, CallRecord] = dataclasses.field(
      default_factory=dict
  )


ModelStatusByOutputFormatType = dict[OutputFormatType, ModelStatus]


class ModelCacheManagerStatus(str, enum.Enum):
  """Current operational status of the model cache manager."""

  INITIALIZING = "INITIALIZING"
  CACHE_OPTIONS_NOT_FOUND = "CACHE_OPTIONS_NOT_FOUND"
  CACHE_PATH_NOT_FOUND = "CACHE_PATH_NOT_FOUND"
  CACHE_PATH_NOT_WRITABLE = "CACHE_PATH_NOT_WRITABLE"
  DISABLED = "DISABLED"
  WORKING = "WORKING"


class QueryCacheManagerStatus(str, enum.Enum):
  """Current operational status of the query cache manager."""

  INITIALIZING = "INITIALIZING"
  CACHE_OPTIONS_NOT_FOUND = "CACHE_OPTIONS_NOT_FOUND"
  CACHE_PATH_NOT_FOUND = "CACHE_PATH_NOT_FOUND"
  CACHE_PATH_NOT_WRITABLE = "CACHE_PATH_NOT_WRITABLE"
  DISABLED = "DISABLED"
  WORKING = "WORKING"


class ProxDashConnectionStatus(str, enum.Enum):
  """Current connection status with the ProxDash service."""

  INITIALIZING = "INITIALIZING"
  DISABLED = "DISABLED"
  API_KEY_NOT_FOUND = "API_KEY_NOT_FOUND"
  API_KEY_NOT_VALID = "API_KEY_NOT_VALID"
  PROXDASH_INVALID_RETURN = "PROXDASH_INVALID_RETURN"
  CONNECTED = "CONNECTED"


ProviderTokenValueMap = dict[str, str]


class StateContainer:
  """Base class for all state objects in the system."""

  pass


@dataclasses.dataclass
class ModelConfigsState(StateContainer):
  """Persisted state for model configuration data."""

  model_registry: ModelRegistry | None = None
  models_by_model_size: ModelSizeMappingType | None = None
  recommended_models: RecommendedModelsMappingType | None = None


@dataclasses.dataclass
class ModelCacheManagerState(StateContainer):
  """Persisted state for the model cache manager."""

  status: ModelCacheManagerStatus | None = None
  cache_options: CacheOptions | None = None


@dataclasses.dataclass
class QueryCacheManagerState(StateContainer):
  """Persisted state for the query cache manager."""

  status: QueryCacheManagerStatus | None = None
  cache_options: CacheOptions | None = None
  shard_count: int | None = 800
  response_per_file: int | None = 200
  cache_response_size: int | None = 40000


@dataclasses.dataclass
class ProxDashConnectionState(StateContainer):
  """Persisted state for ProxDash connection."""

  status: ProxDashConnectionStatus | None = None
  hidden_run_key: str | None = None
  experiment_path: str | None = None
  logging_options: LoggingOptions | None = None
  proxdash_options: ProxDashOptions | None = None
  key_info_from_proxdash: dict | None = None
  connected_experiment_path: str | None = None


@dataclasses.dataclass
class ProviderState(StateContainer):
  """Persisted state for a provider connector (provider-scoped, not per model)."""

  run_type: RunType | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None
  query_cache_manager: QueryCacheManagerState | None = None
  logging_options: LoggingOptions | None = None
  proxdash_connection: ProxDashConnectionState | None = None
  provider_token_value_map: ProviderTokenValueMap | None = None
  keep_raw_provider_response: bool | None = None


@dataclasses.dataclass
class AvailableModelsState(StateContainer):
  """Persisted state for available models discovery."""

  run_type: RunType | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None
  model_configs_instance: ModelConfigsState | None = None
  model_cache_manager: ModelCacheManagerState | None = None
  query_cache_manager: QueryCacheManagerState | None = None
  logging_options: LoggingOptions | None = None
  proxdash_connection: ProxDashConnectionState | None = None
  proxdash_provider_api_keys: ProviderTokenValueMap | None = None
  allow_multiprocessing: bool | None = None
  model_test_timeout: int | None = None
  keep_raw_provider_response: bool | None = None
  providers_with_key: dict[ProviderNameType,
                           ProviderTokenValueMap] | None = (None)
  latest_model_cache_path_used_for_update: str | None = None


@dataclasses.dataclass
class ProxAIClientState(StateContainer):
  """Complete persisted state for a ProxAI client instance."""

  run_type: RunType | None = None
  hidden_run_key: str | None = None
  experiment_path: str | None = None
  root_logging_path: str | None = None
  default_model_cache_path: str | None = None
  platform_used_for_default_model_cache: bool | None = None

  logging_options: LoggingOptions | None = None
  cache_options: CacheOptions | None = None
  proxdash_options: ProxDashOptions | None = None

  model_configs_instance: ModelConfigsState | None = None
  model_configs_requested_from_proxdash: bool | None = None

  registered_model_connectors: dict[OutputFormatType, ProviderState] | None = None
  default_model_cache_manager: ModelCacheManagerState | None = None
  model_cache_manager: ModelCacheManagerState | None = None
  query_cache_manager: QueryCacheManagerState | None = None
  proxdash_connection: ProxDashConnectionState | None = None

  feature_mapping_strategy: FeatureMappingStrategy | None = None
  suppress_provider_errors: bool | None = None
  keep_raw_provider_response: bool | None = None
  allow_multiprocessing: bool | None = None
  model_test_timeout: int | None = None

  available_models_instance: AvailableModelsState | None = None

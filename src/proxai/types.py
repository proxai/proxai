import dataclasses
import datetime
import enum
from abc import ABC
from typing import Any

import pydantic


class RunType(enum.Enum):
  PRODUCTION = 'PRODUCTION'
  TEST = 'TEST'


class CallType(str, enum.Enum):
  GENERATE_TEXT = 'GENERATE_TEXT'
  OTHER = 'OTHER'


ProviderNameType = str
ModelNameType = str
RawProviderModelIdentifierType = str


@dataclasses.dataclass(frozen=True)
class ProviderModelType:
  provider: ProviderNameType
  model: ModelNameType
  provider_model_identifier: RawProviderModelIdentifierType

  def __str__(self):
    return f'({self.provider}, {self.model})'

  def __repr__(self):
    return (
        'ProviderModelType('
        f'provider={self.provider}, '
        f'model={self.model}, '
        f'provider_model_identifier={self.provider_model_identifier})')

  def __lt__(self, other):
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) < str(other)

  def __gt__(self, other):
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) > str(other)

  def __le__(self, other):
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) <= str(other)

  def __ge__(self, other):
    if not isinstance(other, ProviderModelType):
      return NotImplemented
    return str(self) >= str(other)


ProviderModelTupleType = tuple[ProviderNameType, ModelNameType]  # (provider, model) without model_signature
ProviderModelIdentifierType = ProviderModelType | ProviderModelTupleType
StopType = str | list[str]
MessagesType = list[dict[str, str]]


@dataclasses.dataclass
class ProviderModelPricingType:
  per_response_token_cost: float | None = None
  per_query_token_cost: float | None = None


@dataclasses.dataclass
class EndpointFeatureInfoType:
  supported: list[str] = dataclasses.field(default_factory=list)
  best_effort: list[str] = dataclasses.field(default_factory=list)
  not_supported: list[str] = dataclasses.field(default_factory=list)


class FeatureNameType(str, enum.Enum):
  PROMPT = 'prompt'
  MESSAGES = 'messages'
  SYSTEM = 'system'
  MAX_TOKENS = 'max_tokens'
  TEMPERATURE = 'temperature'
  STOP = 'stop'
  WEB_SEARCH = 'web_search'
  RESPONSE_FORMAT_TEXT = 'response_format::text'
  RESPONSE_FORMAT_JSON = 'response_format::json'
  RESPONSE_FORMAT_JSON_SCHEMA = 'response_format::json_schema'
  RESPONSE_FORMAT_PYDANTIC = 'response_format::pydantic'


FeatureListType = list[FeatureNameType]

FeatureListParam = list[str | FeatureNameType]


class ModelSizeType(str, enum.Enum):
  SMALL = 'small'
  MEDIUM = 'medium'
  LARGE = 'large'
  LARGEST = 'largest'

ModelSizeIdentifierType = ModelSizeType | str


@dataclasses.dataclass
class ProviderModelMetadataType:
  call_type: CallType | None = None
  is_featured: bool | None = None
  model_size_tags: list[ModelSizeType] | None = None
  is_default_candidate: bool | None = None
  default_candidate_priority: int | None = None
  tags: list[str] | None = None

FeatureMappingType = dict[FeatureNameType, EndpointFeatureInfoType]


@dataclasses.dataclass
class ProviderModelConfigType:
  provider_model: ProviderModelType | None = None
  pricing: ProviderModelPricingType | None = None
  features: FeatureMappingType | None = None
  metadata: ProviderModelMetadataType | None = None


class ConfigOriginType(enum.Enum):
  BUILT_IN = 'BUILT_IN'
  PROXDASH = 'PROXDASH'

ProviderModelsIdentifierDictType = dict[
    ProviderNameType, tuple[ProviderModelIdentifierType]]

ProviderModelConfigsType = dict[
    ProviderNameType, dict[ModelNameType, ProviderModelConfigType]]
FeaturedModelsType = ProviderModelsIdentifierDictType
ModelsByCallTypeType = dict[CallType, ProviderModelsIdentifierDictType]
ModelsBySizeType = dict[
    ModelSizeType, tuple[ProviderModelIdentifierType]]
DefaultModelPriorityListType = tuple[ProviderModelIdentifierType]


@dataclasses.dataclass
class ModelConfigsSchemaMetadataType:
  version: str | None = None
  released_at: datetime.datetime | None = None
  min_proxai_version: str | None = None
  config_origin: ConfigOriginType | None = None
  release_notes: str | None = None


@dataclasses.dataclass
class ModelConfigsSchemaVersionConfigType:
  provider_model_configs: ProviderModelConfigsType | None = None

  featured_models: FeaturedModelsType | None = None
  models_by_call_type: ModelsByCallTypeType | None = None
  models_by_size: ModelsBySizeType | None = None
  default_model_priority_list: DefaultModelPriorityListType | None = None


@dataclasses.dataclass
class ModelConfigsSchemaType:
  metadata: ModelConfigsSchemaMetadataType | None = None
  version_config: ModelConfigsSchemaVersionConfigType | None = None


@dataclasses.dataclass
class LoggingOptions:
  logging_path: str | None = None
  stdout: bool = False
  hide_sensitive_content: bool = False


class LoggingType(str, enum.Enum):
  QUERY = 'QUERY'
  ERROR = 'ERROR'
  WARNING = 'WARNING'
  INFO = 'INFO'


@dataclasses.dataclass
class CacheOptions:
  cache_path: str | None = None

  unique_response_limit: int | None = 1
  retry_if_error_cached: bool = False
  clear_query_cache_on_connect: bool = False

  disable_model_cache: bool = False
  clear_model_cache_on_connect: bool = False
  model_cache_duration: int | None = None


@dataclasses.dataclass
class ProxDashOptions:
  stdout: bool = False
  hide_sensitive_content: bool = False
  disable_proxdash: bool = False
  api_key: str | None = None
  base_url: str | None = 'https://proxainest-production.up.railway.app'


@dataclasses.dataclass
class SummaryOptions:
  json: bool = True


class FeatureMappingStrategy(str, enum.Enum):
  BEST_EFFORT = 'BEST_EFFORT'
  STRICT = 'STRICT'


@dataclasses.dataclass
class RunOptions:
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


@dataclasses.dataclass
class ResponseFormatPydanticValue:
  class_name: str | None = None
  class_value: type[pydantic.BaseModel] | None = None
  class_json_schema_value: dict[str, Any] | None = None


ResponseFormatValueType = str | dict[str, Any] | ResponseFormatPydanticValue


class ResponseFormatType(str, enum.Enum):
  TEXT = 'TEXT'
  JSON = 'JSON'
  JSON_SCHEMA = 'JSON_SCHEMA'
  PYDANTIC = 'PYDANTIC'


@dataclasses.dataclass
class ResponseFormat:
  value: ResponseFormatValueType | None = None
  type: ResponseFormatType | None = None


ResponseFormatSchema = str | dict[str, Any] | type[pydantic.BaseModel]

@dataclasses.dataclass
class StructuredResponseFormat:
    schema: ResponseFormatSchema | None = None
    type: ResponseFormatType | None = None

ResponseFormatParam = ResponseFormatSchema | StructuredResponseFormat


@dataclasses.dataclass
class QueryRecord:
  call_type: CallType | None = None
  provider_model: ProviderModelType | None = None
  prompt: str | None = None
  system: str | None = None
  messages: MessagesType | None = None
  max_tokens: int | None = None
  temperature: float | None = None
  stop: StopType | None = None
  token_count: int | None = None
  response_format: ResponseFormat | None = None
  web_search: bool | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None
  chosen_endpoint: str | None = None
  hash_value: str | None = None


@dataclasses.dataclass
class PydanticMetadataType:
  class_name: str | None = None
  instance_json_value: dict[str, Any] | None = None


ResponseValue = str | dict[str, Any] | pydantic.BaseModel


class ResponseType(str, enum.Enum):
  TEXT = 'TEXT'
  JSON = 'JSON'
  PYDANTIC = 'PYDANTIC'


@dataclasses.dataclass
class Response:
  value: ResponseValue | None = None
  type: ResponseType | None = None
  pydantic_metadata: PydanticMetadataType | None = None


@dataclasses.dataclass
class QueryResponseRecord:
  response: Response | None = None
  error: str | None = None
  error_traceback: str | None = None
  start_utc_date: datetime.datetime | None = None
  end_utc_date: datetime.datetime | None = None
  local_time_offset_minute: int | None = None
  response_time: datetime.timedelta | None = None
  estimated_cost: int | None = None
  token_count: int | None = None


@dataclasses.dataclass
class CacheRecord:
  query_record: QueryRecord | None = None
  query_responses: list[QueryResponseRecord] = dataclasses.field(
      default_factory=list)
  shard_id: str | None = None
  last_access_time: datetime.datetime | None = None
  call_count: int | None = None


@dataclasses.dataclass
class LightCacheRecord:
  query_record_hash: str | None = None
  query_response_count: int | None = None
  shard_id: int | None = None
  last_access_time: datetime.datetime | None = None
  call_count: int | None = None


class CacheLookFailReason(str, enum.Enum):
  CACHE_NOT_FOUND = 'CACHE_NOT_FOUND'
  CACHE_NOT_MATCHED = 'CACHE_NOT_MATCHED'
  UNIQUE_RESPONSE_LIMIT_NOT_REACHED = 'UNIQUE_RESPONSE_LIMIT_NOT_REACHED'
  PROVIDER_ERROR_CACHED = 'PROVIDER_ERROR_CACHED'


@dataclasses.dataclass
class CacheLookResult:
  query_response: QueryResponseRecord | None = None
  look_fail_reason: CacheLookFailReason | None = None


class ResponseSource(str, enum.Enum):
  CACHE = 'CACHE'
  PROVIDER = 'PROVIDER'


@dataclasses.dataclass
class LoggingRecord:
  query_record: QueryRecord | None = None
  response_record: QueryResponseRecord | None = None
  response_source: ResponseSource | None = None
  look_fail_reason: CacheLookFailReason | None = None


@dataclasses.dataclass
class ModelStatus:
  unprocessed_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  working_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  failed_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  filtered_models: set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  provider_queries: dict[ProviderModelType, LoggingRecord] = (
      dataclasses.field(default_factory=dict))


ModelStatusByCallType = dict[CallType, ModelStatus]


class ModelCacheManagerStatus(str, enum.Enum):
  INITIALIZING = 'INITIALIZING'
  CACHE_OPTIONS_NOT_FOUND = 'CACHE_OPTIONS_NOT_FOUND'
  CACHE_PATH_NOT_FOUND = 'CACHE_PATH_NOT_FOUND'
  CACHE_PATH_NOT_WRITABLE = 'CACHE_PATH_NOT_WRITABLE'
  DISABLED = 'DISABLED'
  WORKING = 'WORKING'


class QueryCacheManagerStatus(str, enum.Enum):
  INITIALIZING = 'INITIALIZING'
  CACHE_OPTIONS_NOT_FOUND = 'CACHE_OPTIONS_NOT_FOUND'
  CACHE_PATH_NOT_FOUND = 'CACHE_PATH_NOT_FOUND'
  CACHE_PATH_NOT_WRITABLE = 'CACHE_PATH_NOT_WRITABLE'
  DISABLED = 'DISABLED'
  WORKING = 'WORKING'


class ProxDashConnectionStatus(str, enum.Enum):
  INITIALIZING = 'INITIALIZING'
  DISABLED = 'DISABLED'
  API_KEY_NOT_FOUND = 'API_KEY_NOT_FOUND'
  API_KEY_NOT_VALID = 'API_KEY_NOT_VALID'
  PROXDASH_INVALID_RETURN = 'PROXDASH_INVALID_RETURN'
  CONNECTED = 'CONNECTED'


class StateContainer(ABC):
    """Base class for all state objects in the system."""
    pass


@dataclasses.dataclass
class ModelConfigsState(StateContainer):
  model_configs_schema: ModelConfigsSchemaType | None = None


@dataclasses.dataclass
class ModelCacheManagerState(StateContainer):
  status: ModelCacheManagerStatus | None = None
  cache_options: CacheOptions | None = None


@dataclasses.dataclass
class QueryCacheManagerState(StateContainer):
  status: QueryCacheManagerStatus | None = None
  cache_options: CacheOptions | None = None
  shard_count: int | None = 800
  response_per_file: int | None = 200
  cache_response_size: int | None = 40000


@dataclasses.dataclass
class ProxDashConnectionState(StateContainer):
  status: ProxDashConnectionStatus | None = None
  hidden_run_key: str | None = None
  experiment_path: str | None = None
  logging_options: LoggingOptions | None = None
  proxdash_options: ProxDashOptions | None = None
  key_info_from_proxdash: dict | None = None
  connected_experiment_path: str | None = None


@dataclasses.dataclass
class ProviderModelState(StateContainer):
  provider_model: ProviderModelType | None = None
  run_type: RunType | None = None
  provider_model_config: ProviderModelConfigType | None = None
  feature_mapping_strategy: FeatureMappingStrategy | None = None
  query_cache_manager: QueryCacheManagerState | None = None
  logging_options: LoggingOptions | None = None
  proxdash_connection: ProxDashConnectionState | None = None


@dataclasses.dataclass
class AvailableModelsState(StateContainer):
  run_type: RunType | None = None
  model_configs_instance: ModelConfigsState | None = None
  model_cache_manager: ModelCacheManagerState | None = None
  logging_options: LoggingOptions | None = None
  proxdash_connection: ProxDashConnectionState | None = None
  allow_multiprocessing: bool | None = None
  model_test_timeout: int | None = None
  providers_with_key: set[str] | None = None
  has_fetched_all_models: bool | None = None
  latest_model_cache_path_used_for_update: str | None = None


@dataclasses.dataclass
class ProxAIClientState(StateContainer):
  run_type: RunType | None = None
  hidden_run_key: str | None = None
  experiment_path: str | None = None
  root_logging_path: str | None = None
  default_model_cache_path: str | None = None
  platform_used_for_default_model_cache: bool | None = None

  logging_options: LoggingOptions | None = None
  cache_options: CacheOptions | None = None
  proxdash_options: ProxDashOptions | None = None

  model_configs: ModelConfigsState | None = None
  model_configs_requested_from_proxdash: bool | None = None

  registered_model_connectors: dict[CallType, ProviderModelState] | None = None
  model_connectors: dict[ProviderModelType, ProviderModelState] | None = None
  default_model_cache_manager: ModelCacheManagerState | None = None
  model_cache_manager: ModelCacheManagerState | None = None
  query_cache_manager: QueryCacheManagerState | None = None
  proxdash_connection: ProxDashConnectionState | None = None

  feature_mapping_strategy: FeatureMappingStrategy | None = None
  suppress_provider_errors: bool | None = None
  allow_multiprocessing: bool | None = None
  model_test_timeout: int | None = None

  available_models: AvailableModelsState | None = None

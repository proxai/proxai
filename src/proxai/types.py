import dataclasses
import datetime
import enum
from typing import Dict, List, Optional, Tuple, Set, Union, Any, Type
from abc import ABC
import pydantic


class RunType(enum.Enum):
  PRODUCTION = 'PRODUCTION'
  TEST = 'TEST'


class CallType(str, enum.Enum):
  GENERATE_TEXT = 'GENERATE_TEXT'


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


ProviderModelTupleType = Tuple[ProviderNameType, ModelNameType]  # (provider, model) without model_signature
ProviderModelIdentifierType = Union[ProviderModelType, ProviderModelTupleType]
StopType = Union[str, List[str]]
MessagesType = List[Dict[str, str]]


@dataclasses.dataclass
class ProviderModelPricingType:
  per_response_token_cost: float
  per_query_token_cost: float


@dataclasses.dataclass
class ProviderModelFeatureType:
  best_effort_features: List[str] = dataclasses.field(default_factory=list)
  not_supported_features: List[str] = dataclasses.field(default_factory=list)


class ModelSizeType(str, enum.Enum):
  SMALL = 'small'
  MEDIUM = 'medium'
  LARGE = 'large'
  LARGEST = 'largest'

ModelSizeIdentifierType = Union[ModelSizeType, str]


@dataclasses.dataclass
class ProviderModelMetadataType:
  call_type: Optional[CallType] = None
  is_featured: Optional[bool] = None
  model_size_tags: Optional[List[ModelSizeType]] = None
  is_default_candidate: Optional[bool] = None
  default_candidate_priority: Optional[int] = None
  tags: Optional[List[str]] = None


@dataclasses.dataclass
class ProviderModelConfigType:
  provider_model: Optional[ProviderModelType] = None
  pricing: Optional[ProviderModelPricingType] = None
  features: Optional[ProviderModelFeatureType] = None
  metadata: Optional[ProviderModelMetadataType] = None


class ConfigOriginType(enum.Enum):
  BUILT_IN = 'BUILT_IN'
  PROXDASH = 'PROXDASH'

ProviderModelsIdentifierDictType = Dict[
    ProviderNameType, Tuple[ProviderModelIdentifierType]]

ProviderModelConfigsType = Dict[
    ProviderNameType, Dict[ModelNameType, ProviderModelConfigType]]
FeaturedModelsType = ProviderModelsIdentifierDictType
ModelsByCallTypeType = Dict[CallType, ProviderModelsIdentifierDictType]
ModelsBySizeType = Dict[
    ModelSizeType, Tuple[ProviderModelIdentifierType]]
DefaultModelPriorityListType = Tuple[ProviderModelIdentifierType]


@dataclasses.dataclass
class ModelConfigsSchemaMetadataType:
  version: Optional[str] = None
  released_at: Optional[datetime.datetime] = None
  min_proxai_version: Optional[str] = None
  config_origin: Optional[ConfigOriginType] = None
  release_notes: Optional[str] = None


@dataclasses.dataclass
class ModelConfigsSchemaVersionConfigType:
  provider_model_configs: Optional[ProviderModelConfigsType] = None

  featured_models: Optional[FeaturedModelsType] = None
  models_by_call_type: Optional[ModelsByCallTypeType] = None
  models_by_size: Optional[ModelsBySizeType] = None
  default_model_priority_list: Optional[DefaultModelPriorityListType] = None


@dataclasses.dataclass
class ModelConfigsSchemaType:
  metadata: Optional[ModelConfigsSchemaMetadataType] = None
  version_config: Optional[ModelConfigsSchemaVersionConfigType] = None


@dataclasses.dataclass
class LoggingOptions:
  logging_path: Optional[str] = None
  stdout: bool = False
  hide_sensitive_content: bool = False


class LoggingType(str, enum.Enum):
  QUERY = 'QUERY'
  ERROR = 'ERROR'
  WARNING = 'WARNING'
  INFO = 'INFO'


@dataclasses.dataclass
class CacheOptions:
  cache_path: Optional[str] = None

  unique_response_limit: Optional[int] = 1
  retry_if_error_cached: bool = False
  clear_query_cache_on_connect: bool = False

  disable_model_cache: bool = False
  clear_model_cache_on_connect: bool = False
  model_cache_duration: Optional[int] = None


@dataclasses.dataclass
class ProxDashOptions:
  stdout: bool = False
  hide_sensitive_content: bool = False
  disable_proxdash: bool = False
  api_key: Optional[str] = None
  base_url: Optional[str] = 'https://proxainest-production.up.railway.app'


@dataclasses.dataclass
class SummaryOptions:
  json: bool = True


class FeatureMappingStrategy(str, enum.Enum):
  BEST_EFFORT = 'BEST_EFFORT'
  OMIT = 'OMIT'
  STRICT = 'STRICT'
  PASSTHROUGH = 'PASSTHROUGH'


@dataclasses.dataclass
class RunOptions:
  run_type: Optional[RunType] = None
  hidden_run_key: Optional[str] = None
  experiment_path: Optional[str] = None
  root_logging_path: Optional[str] = None
  default_model_cache_path: Optional[str] = None
  logging_options: Optional[LoggingOptions] = None
  cache_options: Optional[CacheOptions] = None
  proxdash_options: Optional[ProxDashOptions] = None
  allow_multiprocessing: Optional[bool] = None
  model_test_timeout: Optional[int] = None
  feature_mapping_strategy: Optional[FeatureMappingStrategy] = None
  suppress_provider_errors: Optional[bool] = None


@dataclasses.dataclass
class ResponseFormatPydanticValue:
  class_name: Optional[str] = None
  class_value: Optional[Type[pydantic.BaseModel]] = None
  class_json_schema_value: Optional[Dict[str, Any]] = None


ResponseFormatValueType = Union[
    str,
    Dict[str, Any],
    ResponseFormatPydanticValue
]


class ResponseFormatType(str, enum.Enum):
  TEXT = 'TEXT'
  JSON = 'JSON'
  JSON_SCHEMA = 'JSON_SCHEMA'
  PYDANTIC = 'PYDANTIC'


@dataclasses.dataclass
class ResponseFormat:
  value: Optional[ResponseFormatValueType] = None
  type: Optional[ResponseFormatType] = None


UserDefinedResponseFormatValueType = Union[
    str,
    Dict[str, Any],
    Type[pydantic.BaseModel],
    ResponseFormat
]


@dataclasses.dataclass
class QueryRecord:
  call_type: Optional[CallType] = None
  provider_model: Optional[ProviderModelType] = None
  prompt: Optional[str] = None
  system: Optional[str] = None
  messages: Optional[MessagesType] = None
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  stop: Optional[StopType] = None
  token_count: Optional[int] = None
  response_format: Optional[ResponseFormat] = None
  web_search: Optional[bool] = None
  hash_value: Optional[str] = None


@dataclasses.dataclass
class ResponsePydanticValue:
  class_name: Optional[str] = None
  instance_value: Optional[Type[pydantic.BaseModel]] = None
  instance_json_value: Optional[Dict[str, Any]] = None


ResponseValue = Union[
    str,
    Dict[str, Any],
    ResponsePydanticValue
]


class ResponseType(str, enum.Enum):
  TEXT = 'TEXT'
  JSON = 'JSON'
  PYDANTIC = 'PYDANTIC'


@dataclasses.dataclass
class Response:
  value: Optional[ResponseValue] = None
  type: Optional[ResponseType] = None


@dataclasses.dataclass
class QueryResponseRecord:
  response: Optional[Response] = None
  error: Optional[str] = None
  error_traceback: Optional[str] = None
  start_utc_date: Optional[datetime.datetime] = None
  end_utc_date: Optional[datetime.datetime] = None
  local_time_offset_minute: Optional[int] = None
  response_time: Optional[datetime.timedelta] = None
  estimated_cost: Optional[int] = None
  token_count: Optional[int] = None


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
  CACHE_NOT_FOUND = 'CACHE_NOT_FOUND'
  CACHE_NOT_MATCHED = 'CACHE_NOT_MATCHED'
  UNIQUE_RESPONSE_LIMIT_NOT_REACHED = 'UNIQUE_RESPONSE_LIMIT_NOT_REACHED'
  PROVIDER_ERROR_CACHED = 'PROVIDER_ERROR_CACHED'


@dataclasses.dataclass
class CacheLookResult:
  query_response: Optional[QueryResponseRecord] = None
  look_fail_reason: Optional[CacheLookFailReason] = None


class ResponseSource(str, enum.Enum):
  CACHE = 'CACHE'
  PROVIDER = 'PROVIDER'


@dataclasses.dataclass
class LoggingRecord:
  query_record: Optional[QueryRecord] = None
  response_record: Optional[QueryResponseRecord] = None
  response_source: Optional[ResponseSource] = None
  look_fail_reason: Optional[CacheLookFailReason] = None


@dataclasses.dataclass
class ModelStatus:
  unprocessed_models: Set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  working_models: Set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  failed_models: Set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  filtered_models: Set[ProviderModelType] = dataclasses.field(
      default_factory=set)
  provider_queries: Dict[ProviderModelType, LoggingRecord] = (
      dataclasses.field(default_factory=dict))


ModelStatusByCallType = Dict[CallType, ModelStatus]


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
  model_configs_schema: Optional[ModelConfigsSchemaType] = None


@dataclasses.dataclass
class ModelCacheManagerState(StateContainer):
  status: Optional[ModelCacheManagerStatus] = None
  cache_options: Optional[CacheOptions] = None


@dataclasses.dataclass
class QueryCacheManagerState(StateContainer):
  status: Optional[QueryCacheManagerStatus] = None
  cache_options: Optional[CacheOptions] = None
  shard_count: Optional[int] = 800
  response_per_file: Optional[int] = 200
  cache_response_size: Optional[int] = 40000


@dataclasses.dataclass
class ProxDashConnectionState(StateContainer):
  status: Optional[ProxDashConnectionStatus] = None
  hidden_run_key: Optional[str] = None
  experiment_path: Optional[str] = None
  logging_options: Optional[LoggingOptions] = None
  proxdash_options: Optional[ProxDashOptions] = None
  key_info_from_proxdash: Optional[Dict] = None
  connected_experiment_path: Optional[str] = None


@dataclasses.dataclass
class ProviderModelState(StateContainer):
  provider_model: Optional[ProviderModelType] = None
  run_type: Optional[RunType] = None
  provider_model_config: Optional[ProviderModelConfigType] = None
  feature_mapping_strategy: Optional[FeatureMappingStrategy] = None
  query_cache_manager: Optional[QueryCacheManagerState] = None
  logging_options: Optional[LoggingOptions] = None
  proxdash_connection: Optional[ProxDashConnectionState] = None


@dataclasses.dataclass
class AvailableModelsState(StateContainer):
  run_type: Optional[RunType] = None
  model_configs: Optional[ModelConfigsState] = None
  model_cache_manager: Optional[ModelCacheManagerState] = None
  logging_options: Optional[LoggingOptions] = None
  proxdash_connection: Optional[ProxDashConnectionState] = None
  allow_multiprocessing: Optional[bool] = None
  model_test_timeout: Optional[int] = None
  providers_with_key: Optional[Set[str]] = None
  has_fetched_all_models: Optional[bool] = None
  latest_model_cache_path_used_for_update: Optional[str] = None

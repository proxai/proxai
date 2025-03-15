import dataclasses
import datetime
import enum
from typing import Dict, List, Optional, Tuple, Set, Union


class RunType(enum.Enum):
  PRODUCTION = 'PRODUCTION'
  TEST = 'TEST'


class CallType(str, enum.Enum):
  GENERATE_TEXT = 'GENERATE_TEXT'


@dataclasses.dataclass(frozen=True)
class ProviderModelType:
  provider: str
  model: str
  provider_model_identifier: str

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


ProviderModelTupleType = Tuple[str, str]  # (provider, model) without model_signature
ProviderModelIdentifierType = Union[ProviderModelType, ProviderModelTupleType]
StopType = Union[str, List[str]]
MessagesType = List[Dict[str, str]]


@dataclasses.dataclass
class ProviderModelPricingType:
  per_response_token_cost: float
  per_query_token_cost: float


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
  duration: Optional[int] = None
  retry_if_error_cached: bool = False
  clear_query_cache_on_connect: bool = False
  clear_model_cache_on_connect: bool = False


@dataclasses.dataclass
class ProxDashOptions:
  stdout: bool = False
  hide_sensitive_content: bool = False
  disable_proxdash: bool = False


@dataclasses.dataclass
class SummaryOptions:
  json: bool = True


@dataclasses.dataclass
class RunOptions:
  run_type: Optional[RunType] = None
  logging_options: Optional[LoggingOptions] = None
  cache_options: Optional[CacheOptions] = None
  proxdash_options: Optional[ProxDashOptions] = None
  allow_multiprocessing: Optional[bool] = None
  strict_feature_test: Optional[bool] = None
  suppress_provider_errors: Optional[bool] = None


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
  hash_value: Optional[str] = None


@dataclasses.dataclass
class QueryResponseRecord:
  response: Optional[str] = None
  error: Optional[str] = None
  error_traceback: Optional[str] = None
  start_utc_date: Optional[datetime.datetime] = None
  end_utc_date: Optional[datetime.datetime] = None
  local_time_offset_minute: Optional[int] = None
  response_time: Optional[datetime.timedelta] = None
  estimated_cost: Optional[int] = None


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
  provider_queries: List[LoggingRecord] = (
      dataclasses.field(default_factory=list))


class ProxDashConnectionStatus(str, enum.Enum):
  INITIALIZING = 'INITIALIZING'
  DISABLED = 'DISABLED'
  API_KEY_NOT_FOUND = 'API_KEY_NOT_FOUND'
  API_KEY_NOT_VALID = 'API_KEY_NOT_VALID'
  PROXDASH_INVALID_RETURN = 'PROXDASH_INVALID_RETURN'
  CONNECTED = 'CONNECTED'


@dataclasses.dataclass
class ProxDashConnectionState:
  status: Optional[ProxDashConnectionStatus] = None
  hidden_run_key: Optional[str] = None
  api_key: Optional[str] = None
  experiment_path: Optional[str] = None
  logging_options: Optional[LoggingOptions] = None
  proxdash_options: Optional[ProxDashOptions] = None
  key_info_from_proxdash: Optional[Dict] = None
  connected_experiment_path: Optional[str] = None


@dataclasses.dataclass
class ModelInitState:
  provider_model: Optional[ProviderModelType] = None
  run_type: Optional[RunType] = None
  strict_feature_test: Optional[bool] = None
  logging_options: Optional[LoggingOptions] = None
  proxdash_connection_state: Optional[ProxDashConnectionState] = None

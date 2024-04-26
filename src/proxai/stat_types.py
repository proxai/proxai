import dataclasses
from typing import Dict, List, Optional
import proxai.types as types

@dataclasses.dataclass
class BaseProviderStats:
  total_queries: int = 0
  total_successes: int = 0
  total_fails: int = 0

  total_token_count: int = 0
  total_query_token_count: float = 0.0
  total_response_token_count: int = 0

  total_response_time: float = 0.0
  avr_response_time: float = 0.0

  estimated_price: float = 0.0


@dataclasses.dataclass
class BaseCacheStats:
  total_cache_hit: int = 0

  saved_token_count: int = 0
  saved_query_token_count: float = 0.0
  saved_response_token_count: int = 0

  saved_total_response_time: float = 0.0
  saved_avr_response_time: float = 0.0

  saved_estimated_price: float = 0.0

  total_cache_look_fail_reasons: Dict[types.CacheLookFailReason, int] = (
      dataclasses.field(default_factory=dict))


@dataclasses.dataclass
class ModelStats:
  model: Optional[types.ModelType] = None
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None


@dataclasses.dataclass
class ProviderStats:
  provider: Optional[types.Provider] = None
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  model_stats: Optional[Dict[types.ModelType, ModelStats]] = None


@dataclasses.dataclass
class ConnectStats:
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  provider_stats: Optional[Dict[types.Provider, ProviderStats]] = None


@dataclasses.dataclass
class GlobalStats:
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  last_connect_stats: Optional[List[ConnectStats]] = None

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

  def __add__(self, other):
    return BaseProviderStats(
        total_queries=self.total_queries + other.total_queries,
        total_successes=self.total_successes + other.total_successes,
        total_fails=self.total_fails + other.total_fails,
        total_token_count=self.total_token_count + other.total_token_count,
        total_query_token_count=(
            self.total_query_token_count + other.total_query_token_count),
        total_response_token_count=(
            self.total_response_token_count + other.total_response_token_count),
        total_response_time=self.total_response_time + other.total_response_time,
        avr_response_time=(
            (self.total_response_time + other.total_response_time)
            / (self.total_successes + other.total_successes)),
        estimated_price=self.estimated_price + other.estimated_price)

  def __sub__(self, other):
    return BaseProviderStats(
        total_queries=self.total_queries - other.total_queries,
        total_successes=self.total_successes - other.total_successes,
        total_fails=self.total_fails - other.total_fails,
        total_token_count=self.total_token_count - other.total_token_count,
        total_query_token_count=(
            self.total_query_token_count - other.total_query_token_count),
        total_response_token_count=(
            self.total_response_token_count - other.total_response_token_count),
        total_response_time=self.total_response_time - other.total_response_time,
        avr_response_time=(
            (self.total_response_time - other.total_response_time)
            / (self.total_successes - other.total_successes)),
        estimated_price=self.estimated_price - other.estimated_price)


@dataclasses.dataclass
class BaseCacheStats:
  total_cache_hit: int = 0
  total_success_return: int = 0
  total_fail_return: int = 0

  saved_token_count: int = 0
  saved_query_token_count: float = 0.0
  saved_response_token_count: int = 0

  saved_total_response_time: float = 0.0
  saved_avr_response_time: float = 0.0

  saved_estimated_price: float = 0.0

  total_cache_look_fail_reasons: Dict[types.CacheLookFailReason, int] = (
      dataclasses.field(default_factory=dict))

  def __add__(self, other):
    return BaseCacheStats(
        total_cache_hit=self.total_cache_hit + other.total_cache_hit,
        total_success_return=(
            self.total_success_return + other.total_success_return),
        total_fail_return=self.total_fail_return + other.total_fail_return,
        saved_token_count=self.saved_token_count + other.saved_token_count,
        saved_query_token_count=(
            self.saved_query_token_count + other.saved_query_token_count),
        saved_response_token_count=(
            self.saved_response_token_count + other.saved_response_token_count),
        saved_total_response_time=(
            self.saved_total_response_time + other.saved_total_response_time),
        saved_avr_response_time=(
            (self.total_success_return + other.total_success_return)
            / (self.total_cache_hit + other.total_cache_hit)),
        saved_estimated_price=(
            self.saved_estimated_price + other.saved_estimated_price),
        total_cache_look_fail_reasons={
            k: (self.total_cache_look_fail_reasons.get(k, 0)
                + other.total_cache_look_fail_reasons.get(k, 0))
            for k in (set(self.total_cache_look_fail_reasons)
                      | set(other.total_cache_look_fail_reasons))})

  def __sub__(self, other):
    return BaseCacheStats(
        total_cache_hit=self.total_cache_hit - other.total_cache_hit,
        total_success_return=(
            self.total_success_return - other.total_success_return),
        total_fail_return=self.total_fail_return - other.total_fail_return,
        saved_token_count=self.saved_token_count - other.saved_token_count,
        saved_query_token_count=(
            self.saved_query_token_count - other.saved_query_token_count),
        saved_response_token_count=(
            self.saved_response_token_count - other.saved_response_token_count),
        saved_total_response_time=(
            self.saved_total_response_time - other.saved_total_response_time),
        saved_avr_response_time=(
            (self.total_success_return - other.total_success_return)
            / (self.total_cache_hit - other.total_cache_hit)),
        saved_estimated_price=(
            self.saved_estimated_price - other.saved_estimated_price),
        total_cache_look_fail_reasons={
            k: (self.total_cache_look_fail_reasons.get(k, 0)
                - other.total_cache_look_fail_reasons.get(k, 0))
            for k in (set(self.total_cache_look_fail_reasons)
                      | set(other.total_cache_look_fail_reasons))})


@dataclasses.dataclass
class ModelStats:
  model: Optional[types.ModelType] = None
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None

  def __add__(self, other):
    return ModelStats(
        model=self.model,
        provider_stats=self.provider_stats + other.provider_stats,
        cache_stats=self.cache_stats + other.cache_stats)

  def __sub__(self, other):
    return ModelStats(
        model=self.model,
        provider_stats=self.provider_stats - other.provider_stats,
        cache_stats=self.cache_stats - other.cache_stats)


@dataclasses.dataclass
class ProviderStats:
  provider: Optional[types.Provider] = None
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  model_stats: Optional[Dict[types.ModelType, ModelStats]] = None

  def __add__(self, other):
    return ProviderStats(
        provider=self.provider,
        provider_stats=self.provider_stats + other.provider_stats,
        cache_stats=self.cache_stats + other.cache_stats,
        model_stats={
            k: self.model_stats.get(k, ModelStats())
            + other.model_stats.get(k, ModelStats())
            for k in (set(self.model_stats) | set(other.model_stats))})

  def __sub__(self, other):
    return ProviderStats(
        provider=self.provider,
        provider_stats=self.provider_stats - other.provider_stats,
        cache_stats=self.cache_stats - other.cache_stats,
        model_stats={
            k: self.model_stats.get(k, ModelStats())
            - other.model_stats.get(k, ModelStats())
            for k in (set(self.model_stats) | set(other.model_stats))})


@dataclasses.dataclass
class ConnectStats:
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  provider_stats: Optional[Dict[types.Provider, ProviderStats]] = None
  def __add__(self, other):
    return ConnectStats(
        provider_stats=self.provider_stats + other.provider_stats,
        cache_stats=self.cache_stats + other.cache_stats,
        provider_stats={
            k: self.provider_stats.get(k, ProviderStats())
            + other.provider_stats.get(k, ProviderStats())
            for k in (set(self.provider_stats) | set(other.provider_stats))})

  def __sub__(self, other):
    return ConnectStats(
        provider_stats=self.provider_stats - other.provider_stats,
        cache_stats=self.cache_stats - other.cache_stats,
        provider_stats={
            k: self.provider_stats.get(k, ProviderStats())
            - other.provider_stats.get(k, ProviderStats())
            for k in (set(self.provider_stats) | set(other.provider_stats))})


@dataclasses.dataclass
class GlobalStats:
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None
  last_connect_stats: Optional[List[ConnectStats]] = None

  def __add__(self, other):
    return GlobalStats(
        provider_stats=self.provider_stats + other.provider_stats,
        cache_stats=self.cache_stats + other.cache_stats,
        last_connect_stats=(
            self.last_connect_stats + other.last_connect_stats))

  def __sub__(self, other):
    return GlobalStats(
        provider_stats=self.provider_stats - other.provider_stats,
        cache_stats=self.cache_stats - other.cache_stats,
        last_connect_stats=(
            self.last_connect_stats[:-len(other.last_connect_stats)]))

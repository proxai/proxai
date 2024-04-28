import copy
import dataclasses
import enum
from typing import Dict, List, Optional
import proxai.types as types


@dataclasses.dataclass
class BaseProviderStats:
  total_queries: int = 0
  total_successes: int = 0
  total_fails: int = 0

  total_token_count: int = 0
  total_query_token_count: int = 0
  total_response_token_count: int = 0

  _total_response_time: float = 0.0
  _avr_response_time: float = 0.0

  estimated_price: float = 0.0

  def __init__(
      self,
      total_queries: int = 0,
      total_successes: int = 0,
      total_fails: int = 0,
      total_token_count: int = 0,
      total_query_token_count: int = 0,
      total_response_token_count: int = 0,
      total_response_time: float = 0.0,
      estimated_price: float = 0.0,
  ):
    self.total_queries = total_queries
    self.total_successes = total_successes
    self.total_fails = total_fails
    self.total_token_count = total_token_count
    self.total_query_token_count = total_query_token_count
    self.total_response_token_count = total_response_token_count
    self.total_response_time = total_response_time
    self.estimated_price = estimated_price

  @property
  def total_response_time(self):
    return self._total_response_time

  @total_response_time.setter
  def total_response_time(self, value):
    self._total_response_time = value
    if self.total_successes == 0:
      self._avr_response_time = 0.0
    else:
      self._avr_response_time = (
          self.total_response_time / self.total_successes)

  @property
  def avr_response_time(self):
    return self._avr_response_time

  @avr_response_time.setter
  def avr_response_time(self, value):
    raise ValueError('avr_response_time is a calculated property.')

  def __add__(self, other):
    if not isinstance(other, BaseProviderStats):
      raise ValueError(
          'Invalid addition of BaseProviderStats objects.'
          f'Cannot add BaseProviderStats with {type(other)}')
    return BaseProviderStats(
        total_queries=self.total_queries + other.total_queries,
        total_successes=self.total_successes + other.total_successes,
        total_fails=self.total_fails + other.total_fails,
        total_token_count=self.total_token_count + other.total_token_count,
        total_query_token_count=(
            self.total_query_token_count + other.total_query_token_count),
        total_response_token_count=(
            self.total_response_token_count + other.total_response_token_count),
        total_response_time=(
            self.total_response_time + other.total_response_time),
        estimated_price=self.estimated_price + other.estimated_price)

  def __sub__(self, other):
    if not isinstance(other, BaseProviderStats):
      raise ValueError(
          'Invalid subtraction of BaseProviderStats objects.'
          f'Cannot subtract BaseProviderStats with {type(other)}')
    result = BaseProviderStats(
        total_queries=self.total_queries - other.total_queries,
        total_successes=self.total_successes - other.total_successes,
        total_fails=self.total_fails - other.total_fails,
        total_token_count=self.total_token_count - other.total_token_count,
        total_query_token_count=(
            self.total_query_token_count - other.total_query_token_count),
        total_response_token_count=(
            self.total_response_token_count - other.total_response_token_count),
        total_response_time=(
            self.total_response_time - other.total_response_time),
        estimated_price=self.estimated_price - other.estimated_price)
    for key in result.__dict__:
      if getattr(result, key) < 0:
        raise ValueError(
            f'Invalid subtraction of BaseProviderStats objects.'
            f'Negative value for {key}: {getattr(result, key)}')
    return result


@dataclasses.dataclass
class BaseCacheStats:
  total_cache_hit: int = 0
  total_success_return: int = 0
  total_fail_return: int = 0

  saved_token_count: int = 0
  saved_query_token_count: int = 0
  saved_response_token_count: int = 0

  _saved_total_response_time: float = 0.0
  _saved_avr_response_time: float = 0.0

  saved_estimated_price: float = 0.0

  total_cache_look_fail_reasons: Dict[types.CacheLookFailReason, int] = (
      dataclasses.field(default_factory=dict))

  def __init__(
      self,
      total_cache_hit: int = 0,
      total_success_return: int = 0,
      total_fail_return: int = 0,
      saved_token_count: int = 0,
      saved_query_token_count: int = 0,
      saved_response_token_count: int = 0,
      saved_total_response_time: float = 0.0,
      saved_estimated_price: float = 0.0,
      total_cache_look_fail_reasons: Dict[types.CacheLookFailReason, int] = {},
  ):
    self.total_cache_hit = total_cache_hit
    self.total_success_return = total_success_return
    self.total_fail_return = total_fail_return
    self.saved_token_count = saved_token_count
    self.saved_query_token_count = saved_query_token_count
    self.saved_response_token_count = saved_response_token_count
    self.saved_total_response_time = saved_total_response_time
    self.saved_estimated_price = saved_estimated_price
    self.total_cache_look_fail_reasons = total_cache_look_fail_reasons

  @property
  def saved_total_response_time(self):
    return self._saved_total_response_time

  @saved_total_response_time.setter
  def saved_total_response_time(self, value):
    self._saved_total_response_time = value
    if self.total_success_return == 0:
      self._saved_avr_response_time = 0.0
    else:
      self._saved_avr_response_time = (
          self.saved_total_response_time / self.total_success_return)

  @property
  def saved_avr_response_time(self):
    return self._saved_avr_response_time

  @saved_avr_response_time.setter
  def saved_avr_response_time(self, value):
    raise ValueError('saved_avr_response_time is a calculated property.')

  def __add__(self, other):
    if not isinstance(other, BaseCacheStats):
      raise ValueError(
          'Invalid addition of BaseCacheStats objects.'
          f'Cannot add BaseCacheStats with {type(other)}')
    total_cache_look_fail_reasons = copy.deepcopy(
        self.total_cache_look_fail_reasons)
    for k, v in other.total_cache_look_fail_reasons.items():
      if k in total_cache_look_fail_reasons:
        total_cache_look_fail_reasons[k] += v
      else:
        total_cache_look_fail_reasons[k] = v
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
        saved_estimated_price=(
            self.saved_estimated_price + other.saved_estimated_price),
        total_cache_look_fail_reasons=total_cache_look_fail_reasons)

  def __sub__(self, other):
    if not isinstance(other, BaseCacheStats):
      raise ValueError(
          'Invalid subtraction of BaseCacheStats objects.'
          f'Cannot subtract BaseCacheStats with {type(other)}')
    total_cache_look_fail_reasons = copy.deepcopy(
        self.total_cache_look_fail_reasons)
    for k, v in other.total_cache_look_fail_reasons.items():
      if k in total_cache_look_fail_reasons:
        total_cache_look_fail_reasons[k] -= v
      else:
        raise ValueError(
            f'Invalid subtraction of BaseCacheStats objects.'
            f'Key {k} not found in total_cache_look_fail_reasons.')
    for k in list(total_cache_look_fail_reasons.keys()):
      if total_cache_look_fail_reasons[k] <= 0:
        del total_cache_look_fail_reasons[k]

    result = BaseCacheStats(
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
        saved_estimated_price=(
            self.saved_estimated_price - other.saved_estimated_price),
        total_cache_look_fail_reasons=total_cache_look_fail_reasons)

    for key in result.__dict__:
      if isinstance(getattr(result, key), int) and getattr(result, key) < 0:
        raise ValueError(
            f'Invalid subtraction of BaseCacheStats objects.'
            f'Negative value for {key}: {getattr(result, key)}')
      elif isinstance(getattr(result, key), dict):
        for k in getattr(result, key):
          if getattr(result, key)[k] < 0:
            raise ValueError(
                f'Invalid subtraction of BaseCacheStats objects.'
                f'Negative value for {key}[{k}]: {getattr(result, key)[k]}')
    return result


@dataclasses.dataclass
class BaseStats:
  provider_stats: Optional[BaseProviderStats] = None
  cache_stats: Optional[BaseCacheStats] = None

  def _add_base_provider_stats(self, other: BaseProviderStats):
    if self.provider_stats is None:
      self.provider_stats = BaseProviderStats()
    if other:
      self.provider_stats += other

  def _sub_base_provider_stats(self, other: BaseProviderStats):
    if self.provider_stats is None:
      self.provider_stats = BaseProviderStats()
    if other:
      self.provider_stats -= other

  def _add_base_cache_stats(self, other: BaseCacheStats):
    if self.cache_stats is None:
      self.cache_stats = BaseCacheStats()
    if other:
      self.cache_stats += other

  def _sub_base_cache_stats(self, other: BaseCacheStats):
    if self.cache_stats is None:
      self.cache_stats = BaseCacheStats()
    if other:
      self.cache_stats -= other


@dataclasses.dataclass
class ModelStats(BaseStats):
  model: Optional[types.ModelType] = None

  def __add__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, BaseProviderStats):
      result._add_base_provider_stats(other)
    elif isinstance(other, BaseCacheStats):
      result._add_base_cache_stats(other)
    elif isinstance(other, ModelStats):
      if self.model != other.model:
        raise ValueError(
            'Cannot add ModelStats with different model types.'
            f'{self.model} != {other.model}')
      result._add_base_provider_stats(other.provider_stats)
      result._add_base_cache_stats(other.cache_stats)
    else:
      raise ValueError(
          'Invalid addition of ModelStats objects.'
          f'Cannot add ModelStats with {type(other)}')
    return result

  def __sub__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, BaseProviderStats):
      result._sub_base_provider_stats(other)
    elif isinstance(other, BaseCacheStats):
      result._sub_base_cache_stats(other)
    elif isinstance(other, ModelStats):
      if self.model != other.model:
        raise ValueError(
            'Cannot subtract ModelStats with different model types.'
            f'{self.model} != {other.model}')
      result._sub_base_provider_stats(other.provider_stats)
      result._sub_base_cache_stats(other.cache_stats)
    else:
      raise ValueError(
          'Invalid subtraction of ModelStats objects.'
          f'Cannot subtract ModelStats with {type(other)}')
    return result


@dataclasses.dataclass
class ProviderStats(BaseStats):
  provider: Optional[types.Provider] = None
  models: Optional[Dict[types.ModelType, ModelStats]] = None

  def _add_model_stats(self, other: ModelStats):
    self._add_base_provider_stats(other.provider_stats)
    self._add_base_cache_stats(other.cache_stats)
    if self.models is None:
      self.models = {}
    if other.model and other.model not in self.models:
      self.models[other.model] = ModelStats(model=other.model)
    self.models[other.model] += other
    return self.models

  def _sub_model_stats(self, other: ModelStats):
    self._sub_base_provider_stats(other.provider_stats)
    self._sub_base_cache_stats(other.cache_stats)
    if self.models is None:
      self.models = {}
    if other.model and other.model not in self.models:
      self.models[other.model] = ModelStats(model=other.model)
    self.models[other.model] -= other
    return self.models

  def __add__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, ModelStats):
      result._add_model_stats(other)
    elif isinstance(other, ProviderStats):
      if self.provider != other.provider:
        raise ValueError(
            'Cannot add ProviderStats with different provider types.'
            f'{self.provider} != {other.provider}')
      for model_stat in other.models.values():
        result._add_model_stats(model_stat)
    else:
      raise ValueError(
          'Invalid addition of ProviderStats objects.'
          f'Cannot add ProviderStats with {type(other)}')
    return result

  def __sub__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, ModelStats):
      result._sub_model_stats(other)
    elif isinstance(other, ProviderStats):
      if self.provider != other.provider:
        raise ValueError(
            'Cannot subtract ProviderStats with different provider types.'
            f'{self.provider} != {other.provider}')
      for model_stat in other.models.values():
        result._sub_model_stats(model_stat)
      for model in list(result.models.keys()):
        if (result.models[model].provider_stats.total_queries == 0 and
            result.models[model].cache_stats.total_cache_hit == 0):
          del result.models[model]
    else:
      raise ValueError(
          'Invalid subtraction of ProviderStats objects.'
          f'Cannot subtract ProviderStats with {type(other)}')
    return result


class GlobalStatType(str, enum.Enum):
  RUN_TIME = 'run_time'
  SINCE_CONNECT = 'since_connect'


@dataclasses.dataclass
class RunStats(BaseStats):
  providers: Dict[types.Provider, ProviderStats] = dataclasses.field(
      default_factory=dict)

  def _add_models_stats(self, other: ModelStats):
    self._add_base_provider_stats(other.provider_stats)
    self._add_base_cache_stats(other.cache_stats)
    provider, _ = other.model
    if self.providers is None:
      self.providers = {}
    if provider not in self.providers:
      self.providers[provider] = ProviderStats(provider=provider)
    self.providers[provider] += other

  def _sub_models_stats(self, other: ModelStats):
    self._sub_base_provider_stats(other.provider_stats)
    self._sub_base_cache_stats(other.cache_stats)
    provider, _ = other.model
    if self.providers is None:
      self.providers = {}
    if provider not in self.providers:
      self.providers[provider] = ProviderStats(provider=provider)
    self.providers[provider] -= other
    for provider in list(self.providers.keys()):
      if (self.providers[provider].provider_stats.total_queries == 0 and
          self.providers[provider].cache_stats.total_cache_hit == 0):
        del self.providers[provider]

  def _add_provider_stats(self, other: ProviderStats):
    self._add_base_provider_stats(other.provider_stats)
    self._add_base_cache_stats(other.cache_stats)
    if self.providers is None:
      self.providers = {}
    if other.provider not in self.providers:
      self.providers[other.provider] = ProviderStats(provider=other.provider)
    self.providers[other.provider] += other

  def _sub_provider_stats(self, other: ProviderStats):
    self._sub_base_provider_stats(other.provider_stats)
    self._sub_base_cache_stats(other.cache_stats)
    if self.providers is None:
      self.providers = {}
    if other.provider not in self.providers:
      self.providers[other.provider] = ProviderStats(provider=other.provider)
    self.providers[other.provider] -= other
    for provider in list(self.providers.keys()):
      if (self.providers[provider].provider_stats.total_queries == 0 and
          self.providers[provider].cache_stats.total_cache_hit == 0):
        del self.providers[provider]

  def __add__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, ModelStats):
      result._add_models_stats(other)
    elif isinstance(other, ProviderStats):
      result._add_provider_stats(other)
    elif isinstance(other, RunStats):
      for provider_stat in other.providers.values():
        result._add_provider_stats(provider_stat)
    else:
      raise ValueError(
          'Invalid addition of RunStats objects.'
          f'Cannot add RunStats with {type(other)}')
    return result

  def __sub__(self, other):
    result = copy.deepcopy(self)
    if isinstance(other, ModelStats):
      result._sub_models_stats(other)
    elif isinstance(other, ProviderStats):
      result._sub_provider_stats(other)
    elif isinstance(other, RunStats):
      for provider_stat in other.providers.values():
        result._sub_provider_stats(provider_stat)
      for provider in list(result.providers.keys()):
        if (result.providers[provider].provider_stats.total_queries == 0 and
            result.providers[provider].cache_stats.total_cache_hit == 0):
          del result.providers[provider]
    else:
      raise ValueError(
          'Invalid subtraction of RunStats objects.'
          f'Cannot subtract RunStats with {type(other)}')
    return result

import datetime
import traceback
import functools
from typing import Any, Callable, Dict, Optional
import proxai.types as types
from proxai.logging.utils import log_logging_record, log_message, log_proxdash_message
import proxai.caching.query_cache as query_cache
import proxai.type_utils as type_utils
import proxai.stat_types as stats_type
import proxai.serializers.hash_serializer as hash_serializer
import proxai.connections.proxdash as proxdash


class ModelConnector(object):
  model: Optional[types.ModelType] = None
  provider: Optional[str] = None
  provider_model: Optional[str] = None
  run_type: types.RunType
  strict_feature_test: bool = False
  query_cache_manager: Optional[query_cache.QueryCacheManager] = None
  _api: Optional[Any] = None
  _stats: Optional[Dict[str, stats_type.RunStats]] = None
  _logging_options: Optional[types.LoggingOptions] = None
  _get_logging_options: Optional[Dict] = None
  _proxdash_connection: Optional[proxdash.ProxDashConnection] = None
  _get_proxdash_connection: Optional[
      Callable[[bool], proxdash.ProxDashConnection]] = None

  def __init__(
      self,
      model: types.ModelType,
      run_type: types.RunType,
      strict_feature_test: bool = False,
      query_cache_manager: Optional[query_cache.QueryCacheManager] = None,
      stats: Optional[Dict[str, stats_type.RunStats]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[proxdash.ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[bool], proxdash.ProxDashConnection]] = None):
    if logging_options and get_logging_options:
      raise ValueError(
          'logging_options and get_logging_options cannot be set at the same '
          'time.')

    if proxdash_connection and get_proxdash_connection:
      raise ValueError(
          'proxdash_connection and get_proxdash_connection cannot be set at '
          'the same time.')

    self.model = model
    self.provider, self.provider_model = model
    self.run_type = run_type
    self.strict_feature_test = strict_feature_test
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    self.proxdash_connection = proxdash_connection
    self._get_proxdash_connection = get_proxdash_connection
    if query_cache_manager:
      self.query_cache_manager = query_cache_manager
    if stats:
      self._stats = stats

  @property
  def api(self):
    if not self._api:
      if self.run_type == types.RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  @property
  def logging_options(self):
    if self._logging_options:
      return self._logging_options
    if self._get_logging_options:
      return self._get_logging_options()
    return None

  @logging_options.setter
  def logging_options(self, value):
    self._logging_options = value

  @property
  def proxdash_connection(self):
    if self._proxdash_connection:
      return self._proxdash_connection
    if self._get_proxdash_connection:
      return self._get_proxdash_connection()
    return None

  @proxdash_connection.setter
  def proxdash_connection(self, value):
    self._proxdash_connection = value

  def init_model(self):
    raise NotImplementedError

  def init_mock_model(self):
    raise NotImplementedError

  def feature_fail(
      self,
      message: str,
      query_record: Optional[types.QueryRecord] = None):
    if self.strict_feature_test:
      log_message(
          type=types.LoggingType.ERROR,
          logging_options=self.logging_options,
          query_record=query_record,
          message=message)
      raise Exception(message)
    else:
      log_message(
          type=types.LoggingType.WARNING,
          logging_options=self.logging_options,
          query_record=query_record,
          message=message)

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    raise NotImplementedError

  def _get_token_count(
      self, logging_record: types.LoggingRecord) -> int:
    raise NotImplementedError

  def _get_query_token_count(
      self, logging_record: types.LoggingRecord) -> int:
    raise NotImplementedError

  def _get_response_token_count(
      self, logging_record: types.LoggingRecord) -> int:
    raise NotImplementedError

  def _get_estimated_price(
      self, logging_record: types.LoggingRecord) -> float:
    raise NotImplementedError

  def _update_stats(self, logging_record: types.LoggingRecord):
    if not self._stats:
      return
    model = logging_record.query_record.model
    provider_stats = stats_type.BaseProviderStats()
    cache_stats = stats_type.BaseCacheStats()
    if logging_record.response_source == types.ResponseSource.PROVIDER:
      provider_stats.total_queries = 1
      if logging_record.response_record.response:
        provider_stats.total_successes = 1
      else:
        provider_stats.total_fails = 1
      provider_stats.total_token_count = self._get_token_count(
          logging_record=logging_record)
      provider_stats.total_query_token_count = self._get_query_token_count(
          logging_record=logging_record)
      provider_stats.total_response_token_count = (
          self._get_response_token_count(
            logging_record=logging_record))
      provider_stats.total_response_time = (
          logging_record.response_record.response_time.total_seconds())
      provider_stats.estimated_price = self._get_estimated_price(
          logging_record=logging_record)
      provider_stats.total_cache_look_fail_reasons = {
          logging_record.look_fail_reason: 1}
    elif logging_record.response_source == types.ResponseSource.CACHE:
      cache_stats.total_cache_hit = 1
      if logging_record.response_record.response:
        cache_stats.total_success_return = 1
      else:
        cache_stats.total_fail_return = 1
      cache_stats.saved_token_count = self._get_token_count(
          logging_record=logging_record)
      cache_stats.saved_query_token_count = self._get_query_token_count(
          logging_record=logging_record)
      cache_stats.saved_response_token_count = (
          self._get_response_token_count(
            logging_record=logging_record))
      cache_stats.saved_total_response_time = (
          logging_record.response_record.response_time.total_seconds())
      cache_stats.saved_estimated_price = self._get_estimated_price(
          logging_record=logging_record)
    else:
      raise ValueError(
        f'Invalid response source.\n{logging_record.response_source}')

    model_stats = stats_type.ModelStats(
        model=model,
        provider_stats=provider_stats,
        cache_stats=cache_stats)
    self._stats[stats_type.GlobalStatType.RUN_TIME] += model_stats
    self._stats[stats_type.GlobalStatType.SINCE_CONNECT] += model_stats

  def _update_proxdash(self, logging_record: types.LoggingRecord):
    if not self.proxdash_connection:
      return
    try:
      self.proxdash_connection.upload_logging_record(logging_record)
    except Exception as e:
      log_proxdash_message(
          logging_options=self.logging_options,
          message=(
              'ProxDash upload_logging_record failed.\n'
              f'Error message: {e}\n'
              f'Traceback: {traceback.format_exc()}'),
          type=types.LoggingType.ERROR)

  def generate_text_proc(self, query_record: types.QueryRecord) -> dict:
    raise NotImplementedError

  def generate_text(
      self,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = 100,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None,
      model: Optional[types.ModelType] = None,
      use_cache: bool = True,
      unique_response_limit: Optional[int] = None) -> types.LoggingRecord:
    if prompt != None and messages != None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages != None:
      type_utils.check_messages_type(messages)

    query_model = self.model
    if model != None:
      provider, _ = model
      if provider != self.provider:
        raise ValueError(
            'Model provider does not match the connector provider.')
      query_model = model

    start_time = datetime.datetime.now()
    query_record = types.QueryRecord(
        call_type=types.CallType.GENERATE_TEXT,
        model=query_model,
        prompt=prompt,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop)

    updated_query_record = self.feature_check(query_record=query_record)

    look_fail_reason = None
    if self.query_cache_manager and use_cache:
      cache_look_result = None
      response_record = None
      try:
        cache_look_result = self.query_cache_manager.look(
            updated_query_record,
            unique_response_limit=unique_response_limit)
        if cache_look_result.query_response:
          response_record = cache_look_result.query_response
      except Exception as e:
        pass
      if response_record:
        logging_record = types.LoggingRecord(
            query_record=query_record,
            response_record=response_record,
            response_source=types.ResponseSource.CACHE)
        log_logging_record(
            logging_options=self.logging_options,
            logging_record=logging_record)
        self._update_stats(logging_record=logging_record)
        self._update_proxdash(logging_record=logging_record)
        return logging_record
      look_fail_reason = cache_look_result.look_fail_reason
      logging_record = types.LoggingRecord(
          query_record=query_record,
          look_fail_reason=look_fail_reason,
          response_source=types.ResponseSource.CACHE)
      log_logging_record(
          logging_options=self.logging_options,
          logging_record=logging_record)

    response, error, error_traceback = None, None, None
    try:
      response = self.generate_text_proc(query_record=updated_query_record)
    except Exception as e:
      error_traceback = traceback.format_exc()
      error = e

    if response != None:
      query_response_record = functools.partial(
          types.QueryResponseRecord,
          response=response)
    else:
      query_response_record = functools.partial(
          types.QueryResponseRecord,
          error=str(error),
          error_traceback=error_traceback)
    response_record = query_response_record(
        start_time=start_time,
        end_time=datetime.datetime.now(),
        response_time=datetime.datetime.now() - start_time)

    if self.query_cache_manager and use_cache:
      self.query_cache_manager.cache(
          query_record=updated_query_record,
          response_record=response_record,
          unique_response_limit=unique_response_limit)

    logging_record = types.LoggingRecord(
        query_record=query_record,
        response_record=response_record,
        look_fail_reason=look_fail_reason,
        response_source=types.ResponseSource.PROVIDER)
    log_logging_record(
        logging_options=self.logging_options,
        logging_record=logging_record)
    self._update_stats(logging_record=logging_record)
    self._update_proxdash(logging_record=logging_record)
    return logging_record

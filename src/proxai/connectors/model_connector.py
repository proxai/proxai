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
  model: Optional[types.ModelType]
  provider: Optional[str]
  provider_model: Optional[str]
  _run_type: Optional[types.RunType]
  _get_run_type: Optional[Callable[[], types.RunType]]
  _strict_feature_test: Optional[bool]
  _get_strict_feature_test: Optional[Callable[[], bool]]
  _query_cache_manager: Optional[query_cache.QueryCacheManager]
  _get_query_cache_manager: Optional[
      Callable[[], query_cache.QueryCacheManager]]
  _api: Optional[Any]
  _stats: Optional[Dict[str, stats_type.RunStats]]
  _logging_options: Optional[types.LoggingOptions]
  _get_logging_options: Optional[Dict]
  _proxdash_connection: Optional[proxdash.ProxDashConnection]
  _get_proxdash_connection: Optional[
      Callable[[bool], proxdash.ProxDashConnection]]

  def __init__(
      self,
      model: Optional[types.ModelType] = None,
      run_type: Optional[types.RunType] = None,
      get_run_type: Optional[Callable[[], types.RunType]] = None,
      strict_feature_test: Optional[bool] = None,
      get_strict_feature_test: Optional[Callable[[], bool]] = None,
      query_cache_manager: Optional[query_cache.QueryCacheManager] = None,
      get_query_cache_manager: Optional[
          Callable[[], query_cache.QueryCacheManager]] = None,
      stats: Optional[Dict[str, stats_type.RunStats]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[proxdash.ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[bool], proxdash.ProxDashConnection]] = None,
      init_state: Optional[types.ModelInitState] = None):

    if init_state and (
        run_type is not None or
        get_run_type is not None or
        strict_feature_test is not None or
        get_strict_feature_test is not None or
        query_cache_manager is not None or
        get_query_cache_manager is not None or
        stats is not None or
        logging_options is not None or
        get_logging_options is not None or
        proxdash_connection is not None or
        get_proxdash_connection is not None):
      raise ValueError(
          'init_state and other parameters cannot be set at the same time.')

    if init_state and model and (init_state.model != model):
      raise ValueError(
          'init_state.model is not the same as the model parameter.')

    if (not init_state or not init_state.model) and not model:
      raise ValueError('model parameter is required.')

    if init_state:
      model = init_state.model
      run_type = init_state.run_type
      strict_feature_test = init_state.strict_feature_test
      logging_options = init_state.logging_options
      proxdash_connection = proxdash.ProxDashConnection(
          init_state=init_state.proxdash_init_state)

    if run_type is not None and get_run_type is not None:
      raise ValueError(
          'run_type and get_run_type cannot be set at the same time.')

    if strict_feature_test is not None and get_strict_feature_test is not None:
      raise ValueError(
          'strict_feature_test and get_strict_feature_test cannot be set at '
          'the same time.')

    if query_cache_manager is not None and get_query_cache_manager is not None:
      raise ValueError(
          'query_cache_manager and get_query_cache_manager cannot be set at '
          'the same time.')

    if logging_options is not None and get_logging_options is not None:
      raise ValueError(
          'logging_options and get_logging_options cannot be set at the same '
          'time.')

    if proxdash_connection is not None and get_proxdash_connection is not None:
      raise ValueError(
          'proxdash_connection and get_proxdash_connection cannot be set at '
          'the same time.')

    self.model = model
    self.provider, self.provider_model = model
    self.run_type = run_type
    self._get_run_type = get_run_type
    self.strict_feature_test = strict_feature_test
    self._get_strict_feature_test = get_strict_feature_test
    self.query_cache_manager = query_cache_manager
    self._get_query_cache_manager = get_query_cache_manager
    self._stats = stats
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    self.proxdash_connection = proxdash_connection
    self._get_proxdash_connection = get_proxdash_connection

  @property
  def api(self):
    if not getattr(self, '_api', None):
      if self.run_type == types.RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  @property
  def run_type(self):
    if self._run_type:
      return self._run_type
    if self._get_run_type:
      return self._get_run_type()
    return None

  @run_type.setter
  def run_type(self, value):
    self._run_type = value

  @property
  def strict_feature_test(self):
    if self._strict_feature_test:
      return self._strict_feature_test
    if self._get_strict_feature_test:
      return self._get_strict_feature_test()
    return None

  @strict_feature_test.setter
  def strict_feature_test(self, value):
    self._strict_feature_test = value

  @property
  def query_cache_manager(self):
    if self._query_cache_manager:
      return self._query_cache_manager
    if self._get_query_cache_manager:
      return self._get_query_cache_manager()
    return None

  @query_cache_manager.setter
  def query_cache_manager(self, value):
    self._query_cache_manager = value

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

  def _get_estimated_cost(
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
      provider_stats.estimated_cost = (
          logging_record.response_record.estimated_cost)
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
      cache_stats.saved_estimated_cost = (
          logging_record.response_record.estimated_cost)
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

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
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
        response_record.end_utc_date = datetime.datetime.now(
            datetime.timezone.utc)
        response_record.start_utc_date = (
            response_record.end_utc_date - response_record.response_time)
        response_record.local_time_offset_minute = (
            datetime.datetime.now().astimezone().utcoffset().total_seconds()
            // 60) * -1
        logging_record = types.LoggingRecord(
            query_record=query_record,
            response_record=response_record,
            response_source=types.ResponseSource.CACHE)
        logging_record.response_record.estimated_cost = (
            self._get_estimated_cost(logging_record=logging_record))
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
        start_utc_date=start_utc_date,
        end_utc_date=datetime.datetime.now(datetime.timezone.utc),
        local_time_offset_minute=(
            datetime.datetime.now().astimezone().utcoffset().total_seconds()
            // 60) * -1,
        response_time=(
            datetime.datetime.now(datetime.timezone.utc) - start_utc_date))

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
    logging_record.response_record.estimated_cost = (
        self._get_estimated_cost(logging_record=logging_record))
    log_logging_record(
        logging_options=self.logging_options,
        logging_record=logging_record)
    self._update_stats(logging_record=logging_record)
    self._update_proxdash(logging_record=logging_record)
    return logging_record

  def get_init_state(self) -> types.ModelInitState:
    init_state = types.ModelInitState(
        model=self.model,
        run_type=self.run_type,
        strict_feature_test=self.strict_feature_test,
        logging_options=self.logging_options)

    if self.proxdash_connection:
      init_state.proxdash_init_state = self.proxdash_connection.get_init_state()

    return init_state

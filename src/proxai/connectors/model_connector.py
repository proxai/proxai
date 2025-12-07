from __future__ import annotations

import copy
import datetime
import json
import traceback
import functools
import math
from typing import Any, Callable, Dict, List, Optional, Union
import proxai.types as types
import proxai.logging.utils as logging_utils
import proxai.caching.query_cache as query_cache
import proxai.type_utils as type_utils
import proxai.stat_types as stats_type
import proxai.connections.proxdash as proxdash
import proxai.state_controllers.state_controller as state_controller

_PROVIDER_MODEL_STATE_PROPERTY = '_provider_model_state'


class ProviderModelConnector(state_controller.StateControlled):
  _provider_model: Optional[types.ProviderModelType]
  _run_type: Optional[types.RunType]
  _provider_model_config: Optional[types.ProviderModelConfigType]
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
  _provider_model_state: Optional[types.ProviderModelState]

  def __init__(
      self,
      provider_model: Optional[types.ProviderModelType] = None,
      run_type: Optional[types.RunType] = None,
      provider_model_config: Optional[types.ProviderModelConfigType] = None,
      get_run_type: Optional[Callable[[], types.RunType]] = None,
      strict_feature_test: Optional[bool] = None,
      get_strict_feature_test: Optional[Callable[[], bool]] = None,
      query_cache_manager: Optional[query_cache.QueryCacheManager] = None,
      get_query_cache_manager: Optional[
          Callable[[], query_cache.QueryCacheManager]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[proxdash.ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[bool], proxdash.ProxDashConnection]] = None,
      init_state: Optional[types.ProviderModelState] = None,
      stats: Optional[Dict[str, stats_type.RunStats]] = None):
    super().__init__(
        init_state=init_state,
        provider_model=provider_model,
        run_type=run_type,
        provider_model_config=provider_model_config,
        get_run_type=get_run_type,
        strict_feature_test=strict_feature_test,
        get_strict_feature_test=get_strict_feature_test,
        query_cache_manager=query_cache_manager,
        get_query_cache_manager=get_query_cache_manager,
        logging_options=logging_options,
        get_logging_options=get_logging_options,
        proxdash_connection=proxdash_connection,
        get_proxdash_connection=get_proxdash_connection)

    if init_state:
      if init_state.provider_model is None:
        raise ValueError('provider_model needs to be set in init_state.')
      if init_state.provider_model.provider != self.get_provider_name():
        raise ValueError(
            'provider_model needs to be same with the class provider name.\n'
            f'provider_model: {init_state.provider_model}\n'
            f'class provider name: {self.get_provider_name()}')
      self.load_state(init_state)
    else:
      initial_state = self.get_state()

      self._get_run_type = get_run_type
      self._get_strict_feature_test = get_strict_feature_test
      self._get_query_cache_manager = get_query_cache_manager
      self._get_logging_options = get_logging_options
      self._get_proxdash_connection = get_proxdash_connection

      self.provider_model = provider_model
      self.run_type = run_type
      self.provider_model_config = provider_model_config
      self.strict_feature_test = strict_feature_test
      self.query_cache_manager = query_cache_manager
      self.logging_options = logging_options
      self.proxdash_connection = proxdash_connection
      self._stats = stats

      self.handle_changes(initial_state, self.get_state())

  def get_internal_state_property_name(self):
    return _PROVIDER_MODEL_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.ProviderModelState

  def handle_changes(
      self,
      old_state: types.ProviderModelState,
      current_state: types.ProviderModelState):
    result_state = copy.deepcopy(old_state)
    if current_state.provider_model is not None:
      result_state.provider_model = current_state.provider_model
    if current_state.run_type is not None:
      result_state.run_type = current_state.run_type
    if current_state.strict_feature_test is not None:
      result_state.strict_feature_test = current_state.strict_feature_test
    if current_state.logging_options is not None:
      result_state.logging_options = current_state.logging_options
    if current_state.proxdash_connection is not None:
      result_state.proxdash_connection = (
          current_state.proxdash_connection)

    if result_state.provider_model is None:
      raise ValueError(
          'Provider model is not set for both old and new states. '
          'This creates an invalid state change.')

    if result_state.provider_model.provider != self.get_provider_name():
      raise ValueError(
          'Provider needs to be same with the class provider name.\n'
          f'provider_model: {result_state.provider_model}\n'
          f'class provider name: {self.get_provider_name()}')

    if result_state.logging_options is None:
      raise ValueError(
          'Logging options are not set for both old and new states. '
          'This creates an invalid state change.')
    if result_state.proxdash_connection is None:
      raise ValueError(
          'ProxDash connection is not set for both old and new states. '
          'This creates an invalid state change.')

  @property
  def api(self):
    if not getattr(self, '_api', None):
      if self.run_type == types.RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  @api.setter
  def api(self, value):
    raise ValueError('api should not be set directly.')

  @property
  def provider_model(self):
    return self.get_property_value('provider_model')

  @provider_model.setter
  def provider_model(self, value):
    self.set_property_value('provider_model', value)

  @property
  def run_type(self):
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, value):
    self.set_property_value('run_type', value)

  @property
  def provider_model_config(self):
    return self.get_property_value('provider_model_config')

  @provider_model_config.setter
  def provider_model_config(self, value):
    self.set_property_value('provider_model_config', value)

  @property
  def strict_feature_test(self):
    return self.get_property_value('strict_feature_test')

  @strict_feature_test.setter
  def strict_feature_test(self, value):
    self.set_property_value('strict_feature_test', value)

  @property
  def query_cache_manager(self):
    return self.get_state_controlled_property_value('query_cache_manager')

  @query_cache_manager.setter
  def query_cache_manager(self, value):
    self.set_state_controlled_property_value('query_cache_manager', value)

  def query_cache_manager_deserializer(
      self,
      state_value: types.QueryCacheManagerState
  ) -> query_cache.QueryCacheManager:
    return query_cache.QueryCacheManager(init_state=state_value)

  @property
  def logging_options(self):
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, value):
    self.set_property_value('logging_options', value)

  @property
  def proxdash_connection(self):
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value):
    self.set_state_controlled_property_value('proxdash_connection', value)

  def proxdash_connection_deserializer(
      self,
      state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_state=state_value)

  def feature_fail(
      self,
      message: str,
      query_record: Optional[types.QueryRecord] = None):
    if self.strict_feature_test:
      logging_utils.log_message(
          type=types.LoggingType.ERROR,
          logging_options=self.logging_options,
          query_record=query_record,
          message=message)
      raise Exception(message)
    else:
      logging_utils.log_message(
          type=types.LoggingType.WARNING,
          logging_options=self.logging_options,
          query_record=query_record,
          message=message)

  def _extract_json_from_text(text: str) -> dict:
    """Helper function for extracting JSON from text.

    This is useful for some providers that return text with JSON.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    3. Find JSON object pattern in text
    """
    # Strategy 1: Try direct parse
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      pass

    # Strategy 2: Extract from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(code_block_pattern, text)
    for match in matches:
      try:
        return json.loads(match.strip())
      except json.JSONDecodeError:
        continue

    # Strategy 3: Find JSON object pattern
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
      try:
        return json.loads(text[first_brace:last_brace + 1])
      except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response",
        text,
        0)

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    query_record = copy.deepcopy(query_record)
    for feature in self.provider_model_config.features.not_supported_features:
      if getattr(query_record, feature) is not None:
        self.feature_fail(
            message=f'{self.provider_model.model} does not support {feature}.',
            query_record=query_record)
        setattr(query_record, feature, None)
    return query_record

  def get_token_count_estimate(
      self,
      value: Optional[Union[
          str,
          types.Response,
          types.MessagesType]] = None) -> int:
    total = 0
    def _get_token_count_estimate_from_prompt(prompt: str) -> int:
      return math.ceil(max(
          len(prompt) / 4,
          len(prompt.strip().split()) * 1.3))
    if isinstance(value, str):
      total += _get_token_count_estimate_from_prompt(value)
    elif isinstance(value, types.Response):
      if value.type == types.ResponseType.TEXT:
        total += _get_token_count_estimate_from_prompt(value.value)
      elif value.type == types.ResponseType.JSON:
        total += _get_token_count_estimate_from_prompt(json.dumps(value.value))
      elif value.type == types.ResponseType.PYDANTIC:
        if value.value.instance_json_value is not None:
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.value.instance_json_value))
        else:
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.value.instance_value.model_dump()))
      else:
        raise ValueError(f'Invalid response type: {value.type}')
    elif isinstance(value, list):
      total += 2
      for message in value:
        total += _get_token_count_estimate_from_prompt(
            json.dumps(message)) + 4
    else:
      raise ValueError(
        'Invalid value type. Please provide a string, a response value, or a '
        'messages type.\n'
        f'Value type: {type(value)}\n'
        f'Value: {value}')
    return total

  def get_estimated_cost(self, logging_record: types.LoggingRecord):
    query_token_count = logging_record.query_record.token_count
    if type(query_token_count) != int:
      query_token_count = 0
    response_token_count = logging_record.response_record.token_count
    if type(response_token_count) != int:
      response_token_count = 0
    model_pricing_config = self.provider_model_config.pricing
    return math.floor(
        query_token_count * model_pricing_config.per_query_token_cost +
        response_token_count * model_pricing_config.per_response_token_cost)

  def _update_stats(self, logging_record: types.LoggingRecord):
    if getattr(self, '_stats', None) is None:
      return
    provider_stats = stats_type.BaseProviderStats()
    cache_stats = stats_type.BaseCacheStats()
    query_token_count = logging_record.query_record.token_count
    if type(query_token_count) != int:
      query_token_count = 0
    response_token_count = logging_record.response_record.token_count
    if type(response_token_count) != int:
      response_token_count = 0
    if logging_record.response_source == types.ResponseSource.PROVIDER:
      provider_stats.total_queries = 1
      if logging_record.response_record.response:
        provider_stats.total_successes = 1
      else:
        provider_stats.total_fails = 1
      provider_stats.total_token_count = (
          query_token_count + response_token_count)
      provider_stats.total_query_token_count = query_token_count
      provider_stats.total_response_token_count = response_token_count
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
      cache_stats.saved_token_count = (
          query_token_count + response_token_count)
      cache_stats.saved_query_token_count = query_token_count
      cache_stats.saved_response_token_count = response_token_count
      cache_stats.saved_total_response_time = (
          logging_record.response_record.response_time.total_seconds())
      cache_stats.saved_estimated_cost = (
          logging_record.response_record.estimated_cost)
    else:
      raise ValueError(
        f'Invalid response source.\n{logging_record.response_source}')

    provider_model_stats = stats_type.ProviderModelStats(
        provider_model=self.provider_model,
        provider_stats=provider_stats,
        cache_stats=cache_stats)
    self._stats[stats_type.GlobalStatType.RUN_TIME] += provider_model_stats
    self._stats[stats_type.GlobalStatType.SINCE_CONNECT] += provider_model_stats

  def _update_proxdash(self, logging_record: types.LoggingRecord):
    if not self.proxdash_connection:
      return
    try:
      self.proxdash_connection.upload_logging_record(logging_record)
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_connection.proxdash_options,
          message=(
              'ProxDash upload_logging_record failed.\n'
              f'Error message: {e}\n'
              f'Traceback: {traceback.format_exc()}'),
          type=types.LoggingType.ERROR)

  def generate_text(
      self,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = None,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None,
      response_format: Optional[types.ResponseFormat] = None,
      web_search: Optional[bool] = None,
      provider_model: Optional[types.ProviderModelIdentifierType] = None,
      use_cache: bool = True,
      unique_response_limit: Optional[int] = None) -> types.LoggingRecord:
    if prompt != None and messages != None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages != None:
      type_utils.check_messages_type(messages)

    if provider_model is not None:
      if type(provider_model) == types.ProviderModelTupleType:
        provider_model = self.provider_model_config.provider_model
      if provider_model != self.provider_model:
        raise ValueError(
            'provider_model does not match the connector provider_model.'
            f'provider_model: {provider_model}\n'
            f'connector provider_model: {self.provider_model}')

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    query_record = types.QueryRecord(
        call_type=types.CallType.GENERATE_TEXT,
        provider_model=self.provider_model,
        prompt=prompt,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        web_search=web_search,
        token_count=self.get_token_count_estimate(
            value = prompt if prompt is not None else messages))

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
            self.get_estimated_cost(logging_record=logging_record))
        logging_utils.log_logging_record(
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
      logging_utils.log_logging_record(
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
          response=response,
          token_count=self.get_token_count_estimate(value=response))
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
        self.get_estimated_cost(logging_record=logging_record))
    logging_utils.log_logging_record(
        logging_options=self.logging_options,
        logging_record=logging_record)
    self._update_stats(logging_record=logging_record)
    self._update_proxdash(logging_record=logging_record)
    return logging_record

  def get_provider_name(self):
    raise NotImplementedError

  def init_model(self):
    raise NotImplementedError

  def init_mock_model(self):
    raise NotImplementedError

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

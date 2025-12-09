from __future__ import annotations

import copy
import datetime
import re
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
  _feature_mapping_strategy: Optional[types.FeatureMappingStrategy]
  _get_feature_mapping_strategy: Optional[Callable[[], types.FeatureMappingStrategy]]
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
      feature_mapping_strategy: Optional[types.FeatureMappingStrategy] = None,
      get_feature_mapping_strategy: Optional[Callable[[], types.FeatureMappingStrategy]] = None,
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
        feature_mapping_strategy=feature_mapping_strategy,
        get_feature_mapping_strategy=get_feature_mapping_strategy,
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
      self._get_feature_mapping_strategy = get_feature_mapping_strategy
      self._get_query_cache_manager = get_query_cache_manager
      self._get_logging_options = get_logging_options
      self._get_proxdash_connection = get_proxdash_connection

      self.provider_model = provider_model
      self.run_type = run_type
      self.provider_model_config = provider_model_config
      self.feature_mapping_strategy = feature_mapping_strategy
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
    if current_state.feature_mapping_strategy is not None:
      result_state.feature_mapping_strategy = current_state.feature_mapping_strategy
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
  def feature_mapping_strategy(self):
    return self.get_property_value('feature_mapping_strategy')

  @feature_mapping_strategy.setter
  def feature_mapping_strategy(self, value):
    self.set_property_value('feature_mapping_strategy', value)

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

  def _extract_json_from_text(self, text: str) -> dict:
    """Helper function for extracting JSON from text.

    This is useful for some providers that return text with JSON.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    3. Find JSON object pattern in text
    4. Replace single quotes with double quotes (handles Python dict repr)
    """
    text = text.strip()

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
      candidate = text[first_brace:last_brace + 1]
      try:
        return json.loads(candidate)
      except json.JSONDecodeError:
        pass
      # Strategy 4: Replace single quotes with double quotes
      # Only try this if the candidate has no double quotes (pure Python dict style)
      if '"' not in candidate:
        try:
          return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
          pass

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response",
        text,
        0)

  def handle_feature_not_supported(self, query_record: types.QueryRecord):
    if not self.provider_model_config.features.not_supported:
      return

    for feature in self.provider_model_config.features.not_supported:
      if feature.startswith('response_format::'):
        if query_record.response_format is None:
          continue

        response_format_type = feature.split('::')[1]
        if query_record.response_format.type != types.ResponseFormatType(
            response_format_type.upper()):
          continue

        message=f'{self.provider_model.model} does not support {feature}.'
        logging_utils.log_message(
            type=types.LoggingType.ERROR,
            logging_options=self.logging_options,
            query_record=query_record,
            message=message)
        raise Exception(message)

      elif getattr(query_record, feature) is not None:
        message=f'{self.provider_model.model} does not support {feature}.'
        logging_utils.log_message(
            type=types.LoggingType.ERROR,
            logging_options=self.logging_options,
            query_record=query_record,
            message=message)
        raise Exception(message)

  def handle_feature_best_effort(self, query_record: types.QueryRecord):
    if not self.provider_model_config.features.best_effort:
      return

    for feature in self.provider_model_config.features.best_effort:
      if feature.startswith('response_format::'):
        if query_record.response_format is None:
          continue

        response_format_type = feature.split('::')[1]
        if query_record.response_format.type != types.ResponseFormatType(
            response_format_type.upper()):
          continue

        if self.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT:
          message=(
              f'{self.provider_model.model} does not support {feature} '
              'in STRICT mode.')
          logging_utils.log_message(
              type=types.LoggingType.ERROR,
              logging_options=self.logging_options,
              query_record=query_record,
              message=message)
          raise Exception(message)

      elif getattr(query_record, feature) is not None:
        if self.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT:
          message=(
              f'{self.provider_model.model} does not support {feature} '
              'in STRICT mode.')
          logging_utils.log_message(
              type=types.LoggingType.ERROR,
              logging_options=self.logging_options,
              query_record=query_record,
              message=message)
          raise Exception(message)

        if (self.feature_mapping_strategy ==
            types.FeatureMappingStrategy.BEST_EFFORT):
          message = (
              f'{self.provider_model.model} does not support {feature} '
              'in BEST_EFFORT mode.\n'
              'Omitting this feature.')
          logging_utils.log_message(
              type=types.LoggingType.WARNING,
              logging_options=self.logging_options,
              query_record=query_record,
              message=message)
          setattr(query_record, feature, None)

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    """Checks and handles feature support based on feature_mapping_strategy.

    Regular features (e.g., system):
    | Feature Type  | STRICT | BEST_EFFORT | PASSTHROUGH |
    |---------------|--------|-------------|-------------|
    | not_supported | raises | raises      | raises      |
    | best_effort   | raises | omits       | keeps       |

    response_format:: special syntax:
    | Feature Type  | STRICT | BEST_EFFORT | PASSTHROUGH |
    |---------------|--------|-------------|-------------|
    | not_supported | raises | raises      | raises      |
    | best_effort   | raises | keeps       | keeps       |
    """
    query_record = copy.deepcopy(query_record)
    self.handle_feature_not_supported(query_record=query_record)
    self.handle_feature_best_effort(query_record=query_record)
    return query_record

  def _get_system_content_with_schema_guidance(
      self,
      query_record: types.QueryRecord) -> str:
    if (query_record.response_format.type ==
        types.ResponseFormatType.JSON):
      schema_guidance = 'You must respond with valid JSON.'
    elif (query_record.response_format.type ==
        types.ResponseFormatType.JSON_SCHEMA):
      schema_value = query_record.response_format.value
      json_schema_obj = schema_value['json_schema']
      raw_schema = json_schema_obj.get('schema', json_schema_obj)
      schema_guidance = (
          'You must respond with valid JSON that follows this schema:\n'
          f'{json.dumps(raw_schema, indent=2)}')
    elif (query_record.response_format.type ==
          types.ResponseFormatType.PYDANTIC):
      pydantic_class = query_record.response_format.value.class_value
      schema = pydantic_class.model_json_schema()
      schema_guidance = (
          'You must respond with valid JSON that follows this schema:\n'
          f'{json.dumps(schema, indent=2)}')

    if query_record.system is not None:
      return f"{query_record.system}\n\n{schema_guidance}"
    else:
      return schema_guidance

  def _add_response_format_param(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if (query_record.response_format.type ==
        types.ResponseFormatType.JSON):
      return self.json_feature_mapping(
          query_function,
          query_record=query_record)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return self.json_schema_feature_mapping(
          query_function,
          query_record=query_record)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.PYDANTIC):
      return self.pydantic_feature_mapping(
          query_function,
          query_record=query_record)

  def _add_supported_system_and_response_format_params(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    """Supported system and response format parameter addition.

    - Adds system message as is.
    - Adds structured output parameters.
    """
    query_function = self.system_feature_mapping(
        query_function,
        system_message=query_record.system)
    query_function = self._add_response_format_param(
        query_function,
        query_record=query_record)
    return query_function

  def _add_best_effort_system_and_response_format_params(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    """Best effort system and response format parameter addition.

    - Adds system message with schema guidance
    - Ignores structured output parameters.
    """
    system_message = self._get_system_content_with_schema_guidance(
        query_record=query_record)
    query_function = self.system_feature_mapping(
        query_function,
        system_message=system_message)
    return query_function

  def add_system_and_response_format_params(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if query_record.response_format is None:
      return self.system_feature_mapping(
          query_function,
          system_message=query_record.system)

    if query_record.response_format.type == types.ResponseFormatType.TEXT:
      return self.system_feature_mapping(
          query_function,
          system_message=query_record.system)

    feature_id = (
        f'response_format::{query_record.response_format.type.value.lower()}')

    if feature_id in self.provider_model_config.features.supported:
      return self._add_supported_system_and_response_format_params(
          query_function,
          query_record=query_record)
    elif feature_id in self.provider_model_config.features.best_effort:
      if (self.feature_mapping_strategy ==
          types.FeatureMappingStrategy.STRICT):
        raise Exception(
            f'{query_record.provider_model.model} does not support '
            f'{feature_id} in STRICT mode.')

      elif (self.feature_mapping_strategy ==
            types.FeatureMappingStrategy.BEST_EFFORT):
        return self._add_best_effort_system_and_response_format_params(
            query_function,
            query_record=query_record)

      elif (self.feature_mapping_strategy ==
            types.FeatureMappingStrategy.PASSTHROUGH):
        return self._add_supported_system_and_response_format_params(
            query_function,
            query_record=query_record)

    elif feature_id in self.provider_model_config.features.not_supported:
      raise Exception(
          f'{query_record.provider_model.model} does not support {feature_id}.')
    else:
      raise Exception(
          f'{feature_id} not found in provider model config features.\n'
          f'provider model config: {self.provider_model_config}')

  def format_response_from_providers(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if (query_record.response_format is None or
        query_record.response_format.type == types.ResponseFormatType.TEXT):
      return types.Response(
          value=self.format_text_response_from_provider(
              response=response,
              query_record=query_record),
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=self.format_json_response_from_provider(
              response=response,
              query_record=query_record),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=self.format_json_schema_response_from_provider(
              response=response,
              query_record=query_record),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      if (self.feature_mapping_strategy ==
          types.FeatureMappingStrategy.STRICT or
          self.feature_mapping_strategy ==
          types.FeatureMappingStrategy.PASSTHROUGH):
        return types.Response(
            value=types.ResponsePydanticValue(
                class_name=query_record.response_format.value.class_name,
                instance_value=self.format_pydantic_response_from_provider(
                    response=response,
                    query_record=query_record)),
            type=types.ResponseType.PYDANTIC)
      elif (self.feature_mapping_strategy ==
            types.FeatureMappingStrategy.BEST_EFFORT):
        json_value = self.format_json_response_from_provider(
            response=response,
            query_record=query_record)
        pydantic_class = query_record.response_format.value.class_value
        instance = pydantic_class.model_validate(json_value)
        return types.Response(
            value=types.ResponsePydanticValue(
                class_name=query_record.response_format.value.class_name,
                instance_value=instance),
            type=types.ResponseType.PYDANTIC)

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

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    raise NotImplementedError

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise NotImplementedError

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise NotImplementedError

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise NotImplementedError

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    raise NotImplementedError

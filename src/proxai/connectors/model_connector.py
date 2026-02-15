from __future__ import annotations

import copy
import dataclasses
import datetime
import functools
import json
import math
import re
import traceback
from collections.abc import Callable
from functools import reduce
from typing import Any

import proxai.caching.query_cache as query_cache
import proxai.connections.proxdash as proxdash
import proxai.logging.utils as logging_utils
import proxai.state_controllers.state_controller as state_controller
import proxai.type_utils as type_utils
import proxai.types as types
import proxai.chat.chat_session as chat_session
import proxai.connectors.feature_adapter as feature_adapter

_PROVIDER_MODEL_STATE_PROPERTY = '_provider_model_state'


@dataclasses.dataclass
class ProviderModelConnectorParams:
  """Initialization parameters for ProviderModelConnector."""

  provider_model: types.ProviderModelType | None = None
  run_type: types.RunType | None = None
  provider_model_config: types.ProviderModelConfigType | None = None
  feature_mapping_strategy: types.FeatureMappingStrategy | None = None
  query_cache_manager: types.QueryCacheManagerState | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_token_value_map: types.ProviderTokenValueMap | None = None


class ProviderModelConnector(state_controller.StateControlled):
  """Base class for provider-specific model connectors."""

  _provider_model: types.ProviderModelType | None
  _run_type: types.RunType | None
  _provider_model_config: types.ProviderModelConfigType | None
  _feature_mapping_strategy: types.FeatureMappingStrategy | None
  _query_cache_manager: query_cache.QueryCacheManager | None
  _api: Any | None
  _logging_options: types.LoggingOptions | None
  _proxdash_connection: proxdash.ProxDashConnection | None
  _provider_model_state: types.ProviderModelState | None

  _chosen_endpoint_cached_result: dict[str, bool] | None

  PROVIDER_NAME: str
  PROVIDER_API_KEYS: list[str]
  ENDPOINT_CONFIG: dict[str, types.FeatureConfigType]
  ENDPOINT_EXECUTORS: dict[str, Callable | None]

  def __init__(  # noqa: D107
      self,
      init_from_params: ProviderModelConnectorParams | None = None,
      init_from_state: types.ProviderModelState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    self._chosen_endpoint_cached_result = {}

    if init_from_state:
      if init_from_state.provider_model is None:
        raise ValueError('provider_model needs to be set in init_from_state.')
      if init_from_state.provider_model.provider != self.get_provider_name():
        raise ValueError(
            'provider_model needs to be same with the class provider name.\n'
            f'provider_model: {init_from_state.provider_model}\n'
            f'class provider name: {self.get_provider_name()}'
        )
      self.load_state(init_from_state)
    else:
      self.provider_model = init_from_params.provider_model
      self.run_type = init_from_params.run_type
      self.provider_model_config = init_from_params.provider_model_config
      self.feature_mapping_strategy = init_from_params.feature_mapping_strategy
      self.query_cache_manager = init_from_params.query_cache_manager
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.provider_token_value_map = init_from_params.provider_token_value_map
      self._validate_provider_token_value_map()

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    for attr in (
        'PROVIDER_NAME',
        'PROVIDER_API_KEYS',
        'ENDPOINT_CONFIG',
        'ENDPOINT_EXECUTORS',
    ):
      if attr not in cls.__dict__:
        raise TypeError(
          f"{cls.__name__} must define {attr}")

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _PROVIDER_MODEL_STATE_PROPERTY

  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    return types.ProviderModelState

  def _validate_provider_token_value_map(self):
    if self.provider_token_value_map is None:
      raise ValueError('provider_token_value_map needs to be set.')
    for token_name in self.get_required_provider_token_names():
      if token_name not in self.provider_token_value_map:
        raise ValueError(
            f'provider_token_value_map needs to contain {token_name}.\n'
            f'Available token names: {self.provider_token_value_map.keys()}'
        )

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
      self, state_value: types.QueryCacheManagerState
  ) -> query_cache.QueryCacheManager:
    """Deserialize a QueryCacheManager from its state."""
    return query_cache.QueryCacheManager(init_from_state=state_value)

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
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    """Deserialize a ProxDashConnection from its state."""
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def provider_token_value_map(self):
    return self.get_property_value('provider_token_value_map')

  @provider_token_value_map.setter
  def provider_token_value_map(self, value):
    self.set_property_value('provider_token_value_map', value)

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
      # Only try this if candidate has no double quotes (Python dict style)
      if '"' not in candidate:
        try:
          return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
          pass

    raise json.JSONDecodeError(
        "Could not extract valid JSON from response", text, 0
    )

  def get_token_count_estimate(
      self, value: str | types.Response | types.MessagesType | None = None
  ) -> int:
    """Estimate the token count for a prompt, response, or messages."""
    total = 0

    def _get_token_count_estimate_from_prompt(prompt: str) -> int:
      return math.ceil(max(len(prompt) / 4, len(prompt.strip().split()) * 1.3))

    if isinstance(value, str):
      total += _get_token_count_estimate_from_prompt(value)
    elif isinstance(value, types.Response):
      if value.type == types.ResponseType.TEXT:
        total += _get_token_count_estimate_from_prompt(value.value)
      elif value.type == types.ResponseType.JSON:
        total += _get_token_count_estimate_from_prompt(json.dumps(value.value))
      elif value.type == types.ResponseType.PYDANTIC:
        # Try pydantic_metadata.instance_json_value first, then value (instance)
        if (
            value.pydantic_metadata is not None and
            value.pydantic_metadata.instance_json_value is not None
        ):
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.pydantic_metadata.instance_json_value)
          )
        elif value.value is not None:
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.value.model_dump())
          )
      else:
        raise ValueError(f'Invalid response type: {value.type}')
    elif isinstance(value, list):
      total += 2
      for message in value:
        total += _get_token_count_estimate_from_prompt(json.dumps(message)) + 4
    else:
      raise ValueError(
          'Invalid value type. Please provide a string, a response value, or a '
          'messages type.\n'
          f'Value type: {type(value)}\n'
          f'Value: {value}'
      )
    return total

  def get_estimated_cost(self, logging_record: types.LoggingRecord):
    """Calculate the estimated cost for a logging record."""
    query_token_count = logging_record.query_record.token_count
    if not isinstance(query_token_count, int):
      query_token_count = 0
    response_token_count = logging_record.response_record.token_count
    if not isinstance(response_token_count, int):
      response_token_count = 0
    model_pricing_config = self.provider_model_config.pricing

    query_token_cost = model_pricing_config.per_query_token_cost
    if query_token_cost is None:
      query_token_cost = 0
    response_token_cost = model_pricing_config.per_response_token_cost
    if response_token_cost is None:
      response_token_cost = 0

    return math.floor(
        query_token_count * query_token_cost +
        response_token_count * response_token_cost
    )

  def _update_proxdash(self, logging_record: types.LoggingRecord):
    if not self.proxdash_connection:
      return
    try:
      self.proxdash_connection.upload_logging_record(logging_record)
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_connection.proxdash_options, message=(
              'ProxDash upload_logging_record failed.\n'
              f'Error message: {e}\n'
              f'Traceback: {traceback.format_exc()}'
          ), type=types.LoggingType.ERROR
      )

  def find_compatible_endpoint(self, query_record: types.QueryRecord):
    """Find a compatible endpoint for the query record."""
    best_effort_endpoints = []
    for endpoint in sorted(self.ENDPOINT_CONFIG.keys()):
      feature_config = self.ENDPOINT_CONFIG[endpoint]
      adapter = feature_adapter.FeatureAdapter(
          endpoint=endpoint,
          feature_config=feature_config,
      )
      support_level = adapter.get_support_level(query_record=query_record)
      if support_level == types.FeatureSupportType.SUPPORTED:
        return endpoint
      elif support_level == types.FeatureSupportType.BEST_EFFORT:
        best_effort_endpoints.append(endpoint)
    
    if (
        len(best_effort_endpoints) > 0 and
        self.feature_mapping_strategy == types.FeatureMappingStrategy.BEST_EFFORT
    ):
      return sorted(best_effort_endpoints)[0]
    else:
      raise ValueError(
          'No compatible endpoint found for the query record.'
          f'query_record: {query_record}\n'
          'Try to reduce the number of features.'
      )

  def generate_text(
      self, prompt: str | None = None, system: str | None = None,
      messages: types.MessagesType | None = None, max_tokens: int | None = None,
      temperature: float | None = None, stop: types.StopType | None = None,
      response_format: types.ResponseFormat | None = None,
      web_search: bool | None = None,
      provider_model: types.ProviderModelIdentifierType | None = None,
      feature_mapping_strategy: types.FeatureMappingStrategy | None = None,
      use_cache: bool = True, unique_response_limit: int | None = None
  ) -> types.LoggingRecord:
    """Generate text from the model and return a logging record."""
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages is not None:
      type_utils.check_messages_type(messages)

    if response_format is None:
      response_format = types.ResponseFormat(type=types.ResponseFormatType.TEXT)

    if provider_model is not None:
      if isinstance(provider_model, tuple):
        provider_model = self.provider_model_config.provider_model
      if provider_model != self.provider_model:
        raise ValueError(
            'provider_model does not match the connector provider_model.'
            f'provider_model: {provider_model}\n'
            f'connector provider_model: {self.provider_model}'
        )

    if feature_mapping_strategy is None:
      feature_mapping_strategy = self.feature_mapping_strategy

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)

    system_prompt = None
    chat = None
    if messages is not None:
      chat = chat_session.Chat(messages=messages, system_prompt=system)
    else:
      system_prompt = system

    query_record = types.QueryRecord(
        prompt=prompt,
        system_prompt=system_prompt,
        chat=chat,
        parameters=types.ParameterType(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        ),
        tools=[types.Tools.WEB_SEARCH] if web_search else None,
        response_format=response_format,
        connection_options=types.ConnectionOptions(
            provider_model=self.provider_model,
            feature_mapping_strategy=feature_mapping_strategy,
            chosen_endpoint=None,
        ),
    )

    chosen_endpoint = self.find_compatible_endpoint(query_record=query_record)
    query_record.connection_options.chosen_endpoint = chosen_endpoint

    chosen_adapter = feature_adapter.FeatureAdapter(
        endpoint=chosen_endpoint,
        feature_config=self.ENDPOINT_CONFIG[chosen_endpoint],
    )
    modified_query_record = chosen_adapter.adapt_query_record(
        query_record=query_record)
    chosen_executor = self.ENDPOINT_EXECUTORS[chosen_endpoint]
    
    response, error, error_traceback = None, None, None
    try:
      response = chosen_executor(query_record=modified_query_record)
    except Exception as e:
      error_traceback = traceback.format_exc()
      error = e

    if query_record.prompt is not None:
      input_tokens = self.get_token_count_estimate(value=query_record.prompt)
    else:
      input_tokens = self.get_token_count_estimate(value=query_record.chat)
    if response is not None:
      output_tokens = self.get_token_count_estimate(value=response)
    else:
      output_tokens = 0
    end_utc_date=datetime.datetime.now(datetime.timezone.utc)
    local_time_offset_minute = (
        datetime.datetime.now().astimezone().utcoffset().total_seconds() //
        60
    ) * -1

    if response is not None:
      result_record = types.ResultRecord(
          status=types.ResultStatusType.SUCCESS,
          role=types.MessageRoleType.ASSISTANT,
          content=response,
          usage=types.UsageType(
              input_tokens=input_tokens,
              output_tokens=output_tokens,
              total_tokens=input_tokens + output_tokens,
          ),
          timestamp=types.TimeStampType(
              start_utc_date=start_utc_date,
              end_utc_date=end_utc_date,
              local_time_offset_minute=local_time_offset_minute,
              response_time=end_utc_date - start_utc_date
          )
      )
    else:
      result_record = types.ResultRecord(
          status=types.ResultStatusType.SUCCESS,
          role=types.MessageRoleType.ASSISTANT,
          error=str(error),
          error_traceback=error_traceback,
          usage=types.UsageType(
              input_tokens=input_tokens,
              output_tokens=output_tokens,
              total_tokens=input_tokens + output_tokens,
          ),
          timestamp=types.TimeStampType(
              start_utc_date=start_utc_date,
              end_utc_date=end_utc_date,
              local_time_offset_minute=local_time_offset_minute,
              response_time=end_utc_date - start_utc_date
          )
      )

    call_record = types.CallRecord(
        query=query_record,
        result=result_record,
        cache=types.CacheMetadata(
            cache_hit=False,
            response_source=types.ResponseSource.PROVIDER,
            cache_look_fail_reason=None
        )
    )
    return call_record

  def init_model(self):
    """Initialize the provider API client."""
    raise NotImplementedError

  def init_mock_model(self):
    """Initialize a mock API client for testing."""
    raise NotImplementedError

from __future__ import annotations

from typing import Dict, Any, List
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
import proxai.connectors.result_adapter as result_adapter
import proxai.chat.message_content as message_content

_PROVIDER_STATE_PROPERTY = '_provider_state'


@dataclasses.dataclass
class ProviderConnectorParams:
  """Initialization parameters for ProviderConnector."""

  run_type: types.RunType | None = None
  feature_mapping_strategy: types.FeatureMappingStrategy | None = None
  query_cache_manager: types.QueryCacheManagerState | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_token_value_map: types.ProviderTokenValueMap | None = None
  keep_raw_provider_response: bool | None = None


class ProviderConnector(state_controller.StateControlled):
  """Base class for provider-specific connectors (provider-scoped)."""

  _run_type: types.RunType | None
  _feature_mapping_strategy: types.FeatureMappingStrategy | None
  _query_cache_manager: query_cache.QueryCacheManager | None
  _api: Any | None
  _logging_options: types.LoggingOptions | None
  _proxdash_connection: proxdash.ProxDashConnection | None
  _provider_state: types.ProviderState | None
  _keep_raw_provider_response: bool | None

  _chosen_endpoint_cached_result: dict[str, bool] | None

  PROVIDER_NAME: str
  PROVIDER_API_KEYS: list[str]
  ENDPOINT_PRIORITY: list[str]
  ENDPOINT_CONFIG: dict[str, types.FeatureConfigType]
  ENDPOINT_EXECUTORS: dict[str, Callable | None]

  def __init__(  # noqa: D107
      self,
      init_from_params: ProviderConnectorParams | None = None,
      init_from_state: types.ProviderState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    self._chosen_endpoint_cached_result = {}

    if init_from_state:
      self.load_state(init_from_state)
    else:
      self.run_type = init_from_params.run_type
      self.feature_mapping_strategy = init_from_params.feature_mapping_strategy
      self.query_cache_manager = init_from_params.query_cache_manager
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.provider_token_value_map = init_from_params.provider_token_value_map
      self.keep_raw_provider_response = (
          init_from_params.keep_raw_provider_response
      )
      self._validate_provider_token_value_map()

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    for attr in (
        'PROVIDER_NAME',
        'PROVIDER_API_KEYS',
        'ENDPOINT_PRIORITY',
        'ENDPOINT_CONFIG',
        'ENDPOINT_EXECUTORS',
    ):
      if attr not in cls.__dict__:
        raise TypeError(
          f"{cls.__name__} must define {attr}")
    priority_keys = set(cls.__dict__['ENDPOINT_PRIORITY'])
    config_keys = set(cls.__dict__['ENDPOINT_CONFIG'])
    executor_keys = set(cls.__dict__['ENDPOINT_EXECUTORS'])
    if priority_keys != config_keys:
      missing_in_config = priority_keys - config_keys
      extra_in_config = config_keys - priority_keys
      raise ValueError(
          f'{cls.__name__}: ENDPOINT_PRIORITY and ENDPOINT_CONFIG keys '
          f'do not match.\n'
          f'  Missing in ENDPOINT_CONFIG: {missing_in_config or "none"}\n'
          f'  Extra in ENDPOINT_CONFIG: {extra_in_config or "none"}')
    if priority_keys != executor_keys:
      missing_in_executors = priority_keys - executor_keys
      extra_in_executors = executor_keys - priority_keys
      raise ValueError(
          f'{cls.__name__}: ENDPOINT_PRIORITY and ENDPOINT_EXECUTORS keys '
          f'do not match.\n'
          f'  Missing in ENDPOINT_EXECUTORS: {missing_in_executors or "none"}\n'
          f'  Extra in ENDPOINT_EXECUTORS: {extra_in_executors or "none"}')

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _PROVIDER_STATE_PROPERTY

  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    return types.ProviderState

  def _validate_provider_token_value_map(self):
    if self.provider_token_value_map is None:
      raise ValueError('provider_token_value_map needs to be set.')
    for token_name in self.PROVIDER_API_KEYS:
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
  def run_type(self):
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, value):
    self.set_property_value('run_type', value)

  @property
  def feature_mapping_strategy(self) -> types.FeatureMappingStrategy:
    return self.get_property_value('feature_mapping_strategy')

  @feature_mapping_strategy.setter
  def feature_mapping_strategy(self, value):
    self.set_property_value('feature_mapping_strategy', value)

  @property
  def query_cache_manager(self) -> query_cache.QueryCacheManager:
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
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, value):
    self.set_property_value('logging_options', value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
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
  def provider_token_value_map(self) -> types.ProviderTokenValueMap:
    return self.get_property_value('provider_token_value_map')

  @provider_token_value_map.setter
  def provider_token_value_map(self, value):
    self.set_property_value('provider_token_value_map', value)

  @property
  def keep_raw_provider_response(self) -> bool:
    return self.get_property_value('keep_raw_provider_response')

  @keep_raw_provider_response.setter
  def keep_raw_provider_response(self, value):
    self.set_property_value('keep_raw_provider_response', value)

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
      self,
      messages: Dict | chat_session.Chat | List[
          message_content.MessageContent] | None = None
  ) -> int:
    """Estimate the token count for a prompt, response, or messages."""
    if not messages:
      return 0

    total = 0

    def _get_token_count_estimate_from_str(input: str) -> int:
      return math.ceil(max(len(input) / 4, len(input.strip().split()) * 1.3))
    
    if isinstance(messages, dict):
      return _get_token_count_estimate_from_str(json.dumps(messages))
      
    for message in messages:
      if message.type == message_content.ContentType.TEXT:
        total += _get_token_count_estimate_from_str(message.text)
      elif message.type == message_content.ContentType.THINKING:
        total += _get_token_count_estimate_from_str(message.text)
      elif message.type == message_content.ContentType.JSON:
        total += _get_token_count_estimate_from_str(json.dumps(message.json))
      elif message.type == message_content.ContentType.PYDANTIC_INSTANCE:
        total += _get_token_count_estimate_from_str(
            json.dumps(message.pydantic_content.instance_json_value))
      elif message.type == message_content.ContentType.IMAGE:
        pass
      elif message.type == message_content.ContentType.AUDIO:
        pass
      elif message.type == message_content.ContentType.VIDEO:
        pass
      elif message.type == message_content.ContentType.TOOL:
        pass
      else:
        raise ValueError(f'Invalid message type: {message.type}')

    return total

  def get_estimated_cost(
      self,
      call_record: types.CallRecord,
      provider_model_config: types.ProviderModelConfig,
  ):
    """Calculate the estimated cost for a call record."""
    input_token_count = call_record.result.usage.input_tokens
    if input_token_count is None:
      input_token_count = 0
    output_token_count = call_record.result.usage.output_tokens
    if output_token_count is None:
      output_token_count = 0
    model_pricing_config = provider_model_config.pricing

    query_token_cost = model_pricing_config.input_token_cost
    if query_token_cost is None:
      query_token_cost = 0
    response_token_cost = model_pricing_config.output_token_cost
    if response_token_cost is None:
      response_token_cost = 0

    return math.floor(
        input_token_count * query_token_cost +
        output_token_count * response_token_cost
    )

  def _update_proxdash(self, call_record: types.CallRecord):
    if not self.proxdash_connection:
      return
    try:
      self.proxdash_connection.upload_logging_record(call_record)
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_connection.proxdash_options, message=(
              'ProxDash upload_logging_record failed.\n'
              f'Error message: {e}\n'
              f'Traceback: {traceback.format_exc()}'
          ), type=types.LoggingType.ERROR
      )

  def _get_endpoint_support_level(
      self,
      endpoint: str,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> types.FeatureSupportType:
    if endpoint not in self.ENDPOINT_CONFIG:
      raise ValueError(
          f'endpoint {endpoint} is not a valid endpoint.\n'
          f'Valid endpoints: {self.ENDPOINT_CONFIG.keys()}')
    adapter = feature_adapter.FeatureAdapter(
        endpoint=endpoint,
        endpoint_feature_config=self.ENDPOINT_CONFIG[endpoint],
        model_feature_config=provider_model_config.features,
    )
    return adapter.get_query_record_support_level(query_record=query_record)

  def _check_endpoint_support_compatibility(
      self,
      endpoint: str,
      support_level: types.FeatureSupportType,
      query_record: types.QueryRecord,
  ) -> None:
    if support_level == types.FeatureSupportType.NOT_SUPPORTED:
      raise ValueError(
          f'endpoint {endpoint} is not supported.\n'
          f'query_record: {query_record}')
    elif support_level == types.FeatureSupportType.BEST_EFFORT:
      if self.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT:
        raise ValueError(
            f'endpoint {endpoint} is not supported in STRICT mode.\n'
            f'query_record: {query_record}')

  def get_feature_tags_support_level(
      self,
      feature_tags: list[types.FeatureTagType],
      model_feature_config: types.FeatureConfigType,
  ) -> types.FeatureSupportType:
    """Return the best support level for the given feature tags across endpoints.

    Iterates ENDPOINT_PRIORITY, creates a FeatureAdapter per endpoint with both
    endpoint and model configs, and calls adapter.get_feature_tags_support_level.
    Returns SUPPORTED immediately if found, tracks BEST_EFFORT, else NOT_SUPPORTED.
    """
    has_best_effort = False
    for endpoint in self.ENDPOINT_PRIORITY:
      adapter = feature_adapter.FeatureAdapter(
          endpoint=endpoint,
          endpoint_feature_config=self.ENDPOINT_CONFIG[endpoint],
          model_feature_config=model_feature_config,
      )
      support_level = adapter.get_feature_tags_support_level(feature_tags)
      if support_level == types.FeatureSupportType.SUPPORTED:
        return types.FeatureSupportType.SUPPORTED
      elif support_level == types.FeatureSupportType.BEST_EFFORT:
        has_best_effort = True
    if has_best_effort:
      return types.FeatureSupportType.BEST_EFFORT
    return types.FeatureSupportType.NOT_SUPPORTED

  def _find_compatible_endpoint(
      self,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ):
    """Find a compatible endpoint for the query record."""
    best_effort_endpoints = []
    for endpoint in self.ENDPOINT_PRIORITY:
      support_level = self._get_endpoint_support_level(
          endpoint=endpoint,
          query_record=query_record,
          provider_model_config=provider_model_config)
      if support_level == types.FeatureSupportType.SUPPORTED:
        return endpoint
      elif support_level == types.FeatureSupportType.BEST_EFFORT:
        best_effort_endpoints.append(endpoint)
    
    if (
        len(best_effort_endpoints) > 0 and
        self.feature_mapping_strategy == types.FeatureMappingStrategy.BEST_EFFORT
    ):
      return best_effort_endpoints[0]
    else:
      raise ValueError(
          'No compatible endpoint found for the query record.'
          f'query_record: {query_record}\n'
          'Try to reduce the number of features.'
      )

  def _prepare_execution(
      self,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> tuple[Callable, types.QueryRecord]:
    if (query_record.connection_options and
        query_record.connection_options.endpoint is not None):
      chosen_endpoint = query_record.connection_options.endpoint
      support_level = self._get_endpoint_support_level(
          endpoint=chosen_endpoint,
          query_record=query_record,
          provider_model_config=provider_model_config)
      self._check_endpoint_support_compatibility(
          endpoint=chosen_endpoint,
          support_level=support_level,
          query_record=query_record)
    else:
      chosen_endpoint = self._find_compatible_endpoint(
          query_record=query_record,
          provider_model_config=provider_model_config)

    chosen_feature_adapter = feature_adapter.FeatureAdapter(
        endpoint=chosen_endpoint,
        endpoint_feature_config=self.ENDPOINT_CONFIG[chosen_endpoint],
        model_feature_config=provider_model_config.features,
    )
    executor_name = self.ENDPOINT_EXECUTORS[chosen_endpoint]
    chosen_executor = getattr(self, executor_name)

    modified_query_record = chosen_feature_adapter.adapt_query_record(
        query_record=query_record)

    return chosen_executor, chosen_endpoint, modified_query_record

  def _safe_provider_query(
      self,
      execution_function: Callable,
  ) -> tuple[Any, types.ResultRecord]:
    try:
      response = execution_function()
      return response, types.ResultRecord(
          status=types.ResultStatusType.SUCCESS,
          role=types.MessageRoleType.ASSISTANT)
    except Exception as e:
      return None, types.ResultRecord(
          status=types.ResultStatusType.FAILED,
          role=types.MessageRoleType.ASSISTANT,
          error=e,
          error_traceback=traceback.format_exc())

  def _execute_call(
      self,
      chosen_executor: Callable,
      chosen_endpoint: str,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> types.ExecutorResult:
    executor_result: types.ExecutorResult = chosen_executor(
        query_record=query_record)

    if not executor_result.result_record.error:
      chosen_result_adapter = result_adapter.ResultAdapter(
          endpoint=chosen_endpoint,
          endpoint_feature_config=self.ENDPOINT_CONFIG[chosen_endpoint],
          model_feature_config=provider_model_config.features,
      )
      chosen_result_adapter.adapt_result_record(
          query_record=query_record,
          result_record=executor_result.result_record)
    return executor_result

  def _compute_usage(
      self,
      query_record: types.CallRecord,
      result_record: types.ResultRecord,
) -> types.UsageType:
    if query_record.prompt is not None:
      input_tokens = self.get_token_count_estimate(
          messages=[message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text=query_record.prompt)])
    else:
      input_tokens = self.get_token_count_estimate(messages=query_record.chat)
    if result_record.content is not None:
      output_tokens = self.get_token_count_estimate(
          messages=result_record.content)
    else:
      output_tokens = 0
    return types.UsageType(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

  def _compute_timestamp(
      self,
      start_utc_date: datetime.datetime
  ) -> types.TimeStampType:
    end_utc_date=datetime.datetime.now(datetime.timezone.utc)
    local_time_offset_minute = (
        datetime.datetime.now().astimezone().utcoffset().total_seconds() //
        60
    ) * -1
    return types.TimeStampType(
        start_utc_date=start_utc_date,
        end_utc_date=end_utc_date,
        local_time_offset_minute=local_time_offset_minute,
        response_time=end_utc_date - start_utc_date
    )

  def _get_cached_result(
      self,
      query_record: types.QueryRecord,
      connection_options: types.ConnectionOptions,
  ) -> types.ResultRecord | types.CacheLookFailReason | None:
    # NOTE: override_cache_value bypasses the cache lookup so the real
    # provider is called, and then _update_cache() wipes any existing
    # cached bucket for this query and replaces it with the fresh result.
    # NOTE: skip_cache disables cache completely (no read, no write).
    if (connection_options.skip_cache or
        connection_options.override_cache_value or
        not self.query_cache_manager):
      return None
    cache_look_result: types.CacheLookResult | None = None
    try:
      cache_look_result = self.query_cache_manager.look(query_record)
    except Exception:
      pass
    
    if not cache_look_result.result:
      return cache_look_result.cache_look_fail_reason

    result: types.ResultRecord = cache_look_result.result
    result.timestamp.end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    result.timestamp.start_utc_date = (
        result.timestamp.end_utc_date - result.timestamp.response_time
    )
    result.timestamp.local_time_offset_minute = (
        datetime.datetime.now().astimezone().utcoffset().total_seconds() //
        60
    ) * -1
    return result

  def _update_cache(
      self,
      call_record: types.CallRecord,
      connection_options: types.ConnectionOptions,
  ) -> None:
    # NOTE: When override_cache_value is True, the cache manager wipes any
    # existing entry for this query hash and stores the fresh result as a
    # brand-new single-response bucket (call_count reset to 0). With
    # unique_response_limit > 1, subsequent normal calls will refill the
    # bucket from the provider until the limit is reached again.
    # NOTE: skip_cache disables cache completely.
    if connection_options.skip_cache or not self.query_cache_manager:
      return
    self.query_cache_manager.cache(
        query_record=call_record.query,
        result_record=call_record.result,
        override_cache_value=connection_options.override_cache_value)

  def generate(
      self,
      *,
      provider_model: types.ProviderModelType,
      provider_model_config: types.ProviderModelConfig,
      prompt: str | None = None,
      messages: chat_session.Chat | None = None,
      system_prompt: str | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      response_format: types.ResponseFormatType | None = None,
      connection_options: types.ConnectionOptions | None = None,
      connection_metadata: types.ConnectionMetadata | None = None,
  ) -> types.CallRecord:
    """Generate text from the model and return a logging record."""
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be used together')

    if system_prompt is not None and messages is not None:
      raise ValueError(
          'system_prompt and messages cannot be used together. '
          'Please use "system" message in messages to set the system prompt.\n'
          'px.generate(\n'
          '    messages=[\n'
          '        {"role": "system",\n'
          '         "content": "You are a helpful assistant."},\n'
          '        ...])')

    if response_format is None:
      response_format = types.ResponseFormat(type=types.ResponseFormatType.TEXT)

    if connection_options is None:
      connection_options = types.ConnectionOptions()
    
    if connection_metadata is None:
      connection_metadata = types.ConnectionMetadata()

    if provider_model.provider != self.PROVIDER_NAME:
      raise ValueError(
          'provider_model does not match the connector provider.'
          f'provider_model: {provider_model}\n'
          f'connector provider name: {self.PROVIDER_NAME}'
      )

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)

    query_record = types.QueryRecord(
        prompt=prompt,
        chat=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        response_format=response_format,
        connection_options=connection_options,
    )

    (chosen_executor,
     chosen_endpoint,
     modified_query_record) = self._prepare_execution(
         query_record=query_record,
         provider_model_config=provider_model_config)

    cached_result = self._get_cached_result(
        query_record=query_record,
        connection_options=connection_options,
    )
    if isinstance(cached_result, types.ResultRecord):
      connection_metadata.result_source = types.ResultSource.CACHE
      call_record = types.CallRecord(
          query=query_record,
          result=cached_result,
          connection=connection_metadata,
      )
      return call_record
    elif isinstance(cached_result, types.CacheLookFailReason):
      connection_metadata.cache_look_fail_reason = cached_result

    executor_result = self._execute_call(
        chosen_executor=chosen_executor,
        chosen_endpoint=chosen_endpoint,
        query_record=modified_query_record,
        provider_model_config=provider_model_config,
    )
    result_record = executor_result.result_record
    result_record.usage = self._compute_usage(
        query_record=modified_query_record,
        result_record=result_record)
    result_record.timestamp = self._compute_timestamp(
        start_utc_date=start_utc_date)

    connection_metadata.endpoint_used = chosen_endpoint
    connection_metadata.result_source = types.ResultSource.PROVIDER
    connection_metadata.cache_look_fail_reason = None
    debug_info = None
    if executor_result.raw_provider_response is not None:
      debug_info = types.DebugInfo(
          raw_provider_response=executor_result.raw_provider_response
      )
    call_record = types.CallRecord(
        query=query_record,
        result=result_record,
        connection=connection_metadata,
        debug=debug_info,
    )
    estimated_cost = self.get_estimated_cost(
        call_record=call_record,
        provider_model_config=provider_model_config)
    result_record.usage.estimated_cost = estimated_cost

    if call_record.result.status == types.ResultStatusType.FAILED:
      if (not connection_options or
          not connection_options.suppress_provider_errors):
        raise call_record.result.error
      else:
        call_record.result.error = str(call_record.result.error)
    else:
      self._update_cache(
          call_record=call_record,
          connection_options=connection_options)

    return call_record

  def init_model(self):
    """Initialize the provider API client."""
    raise NotImplementedError

  def init_mock_model(self):
    """Initialize a mock API client for testing."""
    raise NotImplementedError

from __future__ import annotations

from typing import Dict, Any, List
import copy
from concurrent.futures import ThreadPoolExecutor
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
import proxai.connectors.adapter_utils as adapter_utils
import proxai.connectors.feature_adapter as feature_adapter
import proxai.connectors.files as files_module
import proxai.connectors.result_adapter as result_adapter
import proxai.chat.message_content as message_content

_PROVIDER_STATE_PROPERTY = '_provider_state'


@dataclasses.dataclass
class ProviderConnectorParams:
  """Initialization parameters for ProviderConnector."""

  run_type: types.RunType | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  query_cache_manager: types.QueryCacheManagerState | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_token_value_map: types.ProviderTokenValueMap | None = None
  debug_options: types.DebugOptions | None = None
  files_manager: files_module.FilesManager | None = None


class ProviderConnector(state_controller.StateControlled):
  """Base class for provider-specific connectors (provider-scoped)."""

  _run_type: types.RunType | None
  _provider_call_options: types.ProviderCallOptions | None
  _query_cache_manager: query_cache.QueryCacheManager | None
  _api: Any | None
  _logging_options: types.LoggingOptions | None
  _proxdash_connection: proxdash.ProxDashConnection | None
  _provider_state: types.ProviderState | None
  _debug_options: types.DebugOptions | None
  _files_manager_instance: files_module.FilesManager | None

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
      self.provider_call_options = init_from_params.provider_call_options
      self.query_cache_manager = init_from_params.query_cache_manager
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.provider_token_value_map = init_from_params.provider_token_value_map
      self.debug_options = init_from_params.debug_options
      self.files_manager_instance = init_from_params.files_manager
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
        raise TypeError(f"{cls.__name__} must define {attr}")
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
          f'  Extra in ENDPOINT_CONFIG: {extra_in_config or "none"}'
      )
    if priority_keys != executor_keys:
      missing_in_executors = priority_keys - executor_keys
      extra_in_executors = executor_keys - priority_keys
      raise ValueError(
          f'{cls.__name__}: ENDPOINT_PRIORITY and ENDPOINT_EXECUTORS keys '
          f'do not match.\n'
          f'  Missing in ENDPOINT_EXECUTORS: {missing_in_executors or "none"}\n'
          f'  Extra in ENDPOINT_EXECUTORS: {extra_in_executors or "none"}'
      )

  def get_internal_state_property_name(self) -> str:
    """Return the name of the internal state property."""
    return _PROVIDER_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
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
  def api(self) -> Any:
    if not getattr(self, '_api', None):
      if self.run_type == types.RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  @api.setter
  def api(self, value: Any) -> None:
    raise ValueError('api should not be set directly.')

  @property
  def run_type(self) -> types.RunType | None:
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, value: types.RunType | None) -> None:
    self.set_property_value('run_type', value)

  @property
  def provider_call_options(self) -> types.ProviderCallOptions:
    return self.get_property_value('provider_call_options')

  @provider_call_options.setter
  def provider_call_options(
      self,
      value: types.ProviderCallOptions | None,
  ) -> None:
    self.set_property_value('provider_call_options', value)

  @property
  def query_cache_manager(self) -> query_cache.QueryCacheManager:
    return self.get_state_controlled_property_value('query_cache_manager')

  @query_cache_manager.setter
  def query_cache_manager(
      self,
      value: query_cache.QueryCacheManager | None,
  ) -> None:
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
  def logging_options(self, value: types.LoggingOptions | None) -> None:
    self.set_property_value('logging_options', value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(
      self,
      value: proxdash.ProxDashConnection | None,
  ) -> None:
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
  def provider_token_value_map(
      self,
      value: types.ProviderTokenValueMap | None,
  ) -> None:
    self.set_property_value('provider_token_value_map', value)

  @property
  def debug_options(self) -> types.DebugOptions:
    return self.get_property_value('debug_options')

  @debug_options.setter
  def debug_options(self, value: types.DebugOptions | None) -> None:
    self.set_property_value('debug_options', value)

  @property
  def files_manager_instance(self) -> files_module.FilesManager:
    return self.get_state_controlled_property_value(
        'files_manager_instance')

  @files_manager_instance.setter
  def files_manager_instance(
      self, value: files_module.FilesManager | None
  ) -> None:
    self.set_state_controlled_property_value(
        'files_manager_instance', value)

  def files_manager_instance_deserializer(
      self, state_value: types.FilesManagerState
  ) -> files_module.FilesManager:
    return files_module.FilesManager(init_from_state=state_value)

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
      self, messages: Dict | chat_session.Chat |
      List[message_content.MessageContent] | None = None
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
            json.dumps(message.pydantic_content.instance_json_value)
        )
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
  ) -> int:
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

  def _upload_call_record_to_proxdash(self, call_record: types.CallRecord):
    if not self.proxdash_connection:
      return
    try:
      self.proxdash_connection.upload_call_record(
          call_record,
          allow_parallel_file_upload=(
              self.provider_call_options.allow_parallel_file_operations
          ),
      )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_connection.proxdash_options, message=(
              'ProxDash upload_call_record failed.\n'
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
          f'Valid endpoints: {self.ENDPOINT_CONFIG.keys()}'
      )
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
          f'query_record: {query_record}'
      )
    elif support_level == types.FeatureSupportType.BEST_EFFORT:
      if self.provider_call_options.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT:
        raise ValueError(
            f'endpoint {endpoint} is not supported in STRICT mode.\n'
            f'query_record: {query_record}'
        )

  def get_feature_tags_support_level(
      self,
      feature_tags: list[types.FeatureTag],
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

  def get_tag_support_level(
      self,
      tags: list,
      resolve_fn: Callable,
      model_feature_config: types.FeatureConfigType,
  ) -> types.FeatureSupportType:
    """Return the best support level for generic tags across endpoints."""
    has_best_effort = False
    for endpoint in self.ENDPOINT_PRIORITY:
      merged = adapter_utils.merge_feature_configs(
          self.ENDPOINT_CONFIG[endpoint],
          model_feature_config,
      )
      levels = [resolve_fn(merged, tag) for tag in tags]
      min_level = min(levels, key=lambda l: adapter_utils.SUPPORT_RANK[l])
      if min_level == types.FeatureSupportType.SUPPORTED:
        return types.FeatureSupportType.SUPPORTED
      elif min_level == types.FeatureSupportType.BEST_EFFORT:
        has_best_effort = True
    if has_best_effort:
      return types.FeatureSupportType.BEST_EFFORT
    return types.FeatureSupportType.NOT_SUPPORTED

  def _raise_no_compatible_endpoint(
      self,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ):
    signature = feature_adapter.FeatureAdapter.get_query_signature(query_record)
    provider = query_record.provider_model.provider
    model = query_record.provider_model.model
    signature_str = json.dumps(signature)

    endpoint_lines = []
    for endpoint in self.ENDPOINT_PRIORITY:
      adapter = feature_adapter.FeatureAdapter(
          endpoint=endpoint,
          endpoint_feature_config=self.ENDPOINT_CONFIG[endpoint],
          model_feature_config=provider_model_config.features,
      )
      details = adapter.get_query_support_details(query_record)
      details_str = json.dumps(details)
      endpoint_lines.append(f"- {endpoint}:\n  {details_str}")

    raise ValueError(
        f"No compatible endpoint found for provider '{provider}' "
        f"model '{model}'. "
        'Try reducing the number of features or using a '
        'different model that supports the requested features.\n'
        f"Query signature:\n  {signature_str}\n"
        f"Endpoints support configs:\n" + '\n'.join(endpoint_lines)
    )

  def _find_compatible_endpoint(
      self,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> str:
    """Find a compatible endpoint for the query record."""
    best_effort_endpoints = []
    for endpoint in self.ENDPOINT_PRIORITY:
      support_level = self._get_endpoint_support_level(
          endpoint=endpoint, query_record=query_record,
          provider_model_config=provider_model_config
      )
      if support_level == types.FeatureSupportType.SUPPORTED:
        return endpoint
      elif support_level == types.FeatureSupportType.BEST_EFFORT:
        best_effort_endpoints.append(endpoint)

    if (
        len(best_effort_endpoints) > 0 and
        self.provider_call_options.feature_mapping_strategy
        == types.FeatureMappingStrategy.BEST_EFFORT
    ):
      return best_effort_endpoints[0]

    self._raise_no_compatible_endpoint(query_record, provider_model_config)

  def _prepare_execution(
      self,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> tuple[Callable, types.QueryRecord]:
    if (
        query_record.connection_options and
        query_record.connection_options.endpoint is not None
    ):
      chosen_endpoint = query_record.connection_options.endpoint
      support_level = self._get_endpoint_support_level(
          endpoint=chosen_endpoint, query_record=query_record,
          provider_model_config=provider_model_config
      )
      self._check_endpoint_support_compatibility(
          endpoint=chosen_endpoint, support_level=support_level,
          query_record=query_record
      )
    else:
      chosen_endpoint = self._find_compatible_endpoint(
          query_record=query_record, provider_model_config=provider_model_config
      )

    chosen_feature_adapter = feature_adapter.FeatureAdapter(
        endpoint=chosen_endpoint,
        endpoint_feature_config=self.ENDPOINT_CONFIG[chosen_endpoint],
        model_feature_config=provider_model_config.features,
    )
    executor_name = self.ENDPOINT_EXECUTORS[chosen_endpoint]
    chosen_executor = getattr(self, executor_name)

    modified_query_record = chosen_feature_adapter.adapt_query_record(
        query_record=query_record
    )

    return chosen_executor, chosen_endpoint, modified_query_record

  def _safe_provider_query(
      self,
      execution_function: Callable,
  ) -> tuple[Any, types.ResultRecord]:
    try:
      response = execution_function()
      return response, types.ResultRecord(
          status=types.ResultStatusType.SUCCESS,
          role=types.MessageRoleType.ASSISTANT
      )
    except Exception as e:
      return None, types.ResultRecord(
          status=types.ResultStatusType.FAILED,
          role=types.MessageRoleType.ASSISTANT, error=e,
          error_traceback=traceback.format_exc()
      )

  def _execute_call(
      self,
      chosen_executor: Callable,
      chosen_endpoint: str,
      query_record: types.QueryRecord,
      provider_model_config: types.ProviderModelConfig,
  ) -> types.ExecutorResult:
    executor_result: types.ExecutorResult = chosen_executor(
        query_record=query_record
    )

    if not executor_result.result_record.error:
      chosen_result_adapter = result_adapter.ResultAdapter(
          endpoint=chosen_endpoint,
          endpoint_feature_config=self.ENDPOINT_CONFIG[chosen_endpoint],
          model_feature_config=provider_model_config.features,
      )
      chosen_result_adapter.adapt_result_record(
          query_record=query_record, result_record=executor_result.result_record
      )
    return executor_result

  def _compute_usage(
      self,
      query_record: types.CallRecord,
      result_record: types.ResultRecord,
  ) -> types.UsageType:
    if query_record.prompt is not None:
      input_tokens = self.get_token_count_estimate(
          messages=[
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT,
                  text=query_record.prompt
              )
          ]
      )
    else:
      input_tokens = self.get_token_count_estimate(messages=query_record.chat)
    if result_record.content is not None:
      output_tokens = self.get_token_count_estimate(
          messages=result_record.content
      )
    else:
      output_tokens = 0
    return types.UsageType(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

  def _compute_timestamp(
      self, start_utc_date: datetime.datetime
  ) -> types.TimeStampType:
    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    local_time_offset_minute = (
        datetime.datetime.now().astimezone().utcoffset().total_seconds() // 60
    ) * -1
    return types.TimeStampType(
        start_utc_date=start_utc_date, end_utc_date=end_utc_date,
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
    if (
        connection_options.skip_cache or
        connection_options.override_cache_value or not self.query_cache_manager
    ):
      return None
    cache_look_result = self.query_cache_manager.look(query_record)
    if not cache_look_result.result:
      return cache_look_result.cache_look_fail_reason

    result: types.ResultRecord = cache_look_result.result
    result.timestamp.end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    result.timestamp.start_utc_date = (
        result.timestamp.end_utc_date - result.timestamp.response_time
    )
    result.timestamp.local_time_offset_minute = (
        datetime.datetime.now().astimezone().utcoffset().total_seconds() // 60
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
        query_record=call_record.query, result_record=call_record.result,
        override_cache_value=connection_options.override_cache_value
    )

  def _reconstruct_pydantic_from_cache(
      self,
      query_record: types.QueryRecord,
      result_record: types.ResultRecord,
  ) -> None:
    """Reconstruct pydantic instances on a cached ResultRecord."""
    if (
        not query_record.output_format or
        query_record.output_format.type != types.OutputFormatType.PYDANTIC or
        not query_record.output_format.pydantic_class
    ):
      return
    pydantic_class = query_record.output_format.pydantic_class
    if result_record.content:
      for mc in result_record.content:
        if (
            mc.pydantic_content and
            mc.pydantic_content.instance_json_value is not None and
            mc.pydantic_content.instance_value is None
        ):
          mc.pydantic_content.instance_value = (
              pydantic_class.model_validate(
                  mc.pydantic_content.instance_json_value
              )
          )
          mc.pydantic_content.class_value = pydantic_class
    if (result_record.output_pydantic is None and result_record.content):
      for mc in reversed(result_record.content):
        if (
            mc.pydantic_content and
            mc.pydantic_content.instance_value is not None
        ):
          result_record.output_pydantic = (mc.pydantic_content.instance_value)
          break

  _MEDIA_CONTENT_TYPES = {
      message_content.ContentType.IMAGE,
      message_content.ContentType.DOCUMENT,
      message_content.ContentType.AUDIO,
      message_content.ContentType.VIDEO,
  }

  def _auto_upload_media(self, query_record: types.QueryRecord):
    """Upload media files to provider File API before execution.

    Called in generate() before _prepare_execution(). Iterates all
    MessageContent objects in chat messages and uploads media files
    (IMAGE, DOCUMENT, AUDIO, VIDEO) that have local content (path
    or data) but no file_id for the current provider.

    Skips content that:
    - Is not a media type (text, json, pydantic, tool)
    - Already has a file_id for this provider (previously uploaded)
    - Has no local content (remote-only references)
    - Is not supported by this provider's File API (MIME type check)

    Mutates MessageContent objects in place by populating
    provider_file_api_ids and provider_file_api_status. This allows
    downstream _to_*_part() methods to use file_id references
    instead of inline base64.

    When run_type=TEST, mock dispatches are used — no real API calls
    are made, but fake file_ids are generated so the file_id
    reference code path is exercised.
    """
    if self.files_manager_instance is None:
      return
    if query_record.chat is None:
      return
    provider = self.PROVIDER_NAME
    pending = []
    seen = set()
    for msg in query_record.chat.messages:
      if isinstance(msg.content, str):
        continue
      for mc in msg.content:
        if id(mc) in seen:
          continue
        seen.add(id(mc))
        if mc.type not in self._MEDIA_CONTENT_TYPES:
          continue
        if (mc.provider_file_api_ids
            and provider in mc.provider_file_api_ids):
          continue
        if mc.path is None and mc.data is None:
          continue
        if not self.files_manager_instance.is_upload_supported(
            mc, provider):
          continue
        pending.append(mc)

    if not pending:
      return
    allow_parallel = (
        self.provider_call_options is not None
        and self.provider_call_options.allow_parallel_file_operations
    )
    if allow_parallel and len(pending) > 1:
      with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futures = [
            pool.submit(
                self.files_manager_instance.upload,
                media=mc, providers=[provider])
            for mc in pending
        ]
        for future in futures:
          try:
            future.result()
          except Exception:
            pass
    else:
      for mc in pending:
        self.files_manager_instance.upload(
            media=mc, providers=[provider])

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
      output_format: types.OutputFormatType | None = None,
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
          '        ...])'
      )

    if output_format is None:
      output_format = types.OutputFormat(type=types.OutputFormatType.TEXT)

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
        output_format=output_format,
        connection_options=connection_options,
    )

    self._auto_upload_media(query_record)

    (chosen_executor, chosen_endpoint,
     modified_query_record) = self._prepare_execution(
         query_record=query_record, provider_model_config=provider_model_config
     )

    cached_result = self._get_cached_result(
        query_record=query_record,
        connection_options=connection_options,
    )
    if isinstance(cached_result, types.ResultRecord):
      self._reconstruct_pydantic_from_cache(
          query_record=query_record, result_record=cached_result
      )
      connection_metadata.result_source = types.ResultSource.CACHE
      call_record = types.CallRecord(
          query=query_record,
          result=cached_result,
          connection=connection_metadata,
      )
      self._upload_call_record_to_proxdash(call_record)
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
        query_record=modified_query_record, result_record=result_record
    )
    result_record.timestamp = self._compute_timestamp(
        start_utc_date=start_utc_date
    )

    connection_metadata.endpoint_used = chosen_endpoint
    connection_metadata.result_source = types.ResultSource.PROVIDER
    connection_metadata.cache_look_fail_reason = None
    debug_info = None
    if (
        executor_result.raw_provider_response is not None and
        self.debug_options is not None and
        self.debug_options.keep_raw_provider_response
    ):
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
        call_record=call_record, provider_model_config=provider_model_config
    )
    result_record.usage.estimated_cost = estimated_cost

    if call_record.result.status == types.ResultStatusType.FAILED:
      if (
          not connection_options or
          not connection_options.suppress_provider_errors
      ):
        self._upload_call_record_to_proxdash(call_record)
        raise call_record.result.error
      else:
        call_record.result.error = str(call_record.result.error)
    else:
      self._update_cache(
          call_record=call_record, connection_options=connection_options
      )

    self._upload_call_record_to_proxdash(call_record)
    return call_record

  def init_model(self) -> Any:
    """Initialize the provider API client."""
    raise NotImplementedError

  def init_mock_model(self) -> Any:
    """Initialize a mock API client for testing."""
    raise NotImplementedError

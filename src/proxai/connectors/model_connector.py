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

_PROVIDER_MODEL_STATE_PROPERTY = '_provider_model_state'


@dataclasses.dataclass
class ProviderModelConnectorParams:
  provider_model: types.ProviderModelType | None = None
  run_type: types.RunType | None = None
  provider_model_config: types.ProviderModelConfigType | None = None
  feature_mapping_strategy: types.FeatureMappingStrategy | None = None
  query_cache_manager: types.QueryCacheManagerState | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None


class ProviderModelConnector(state_controller.StateControlled):
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

  def __init__(
      self,
      init_from_params: ProviderModelConnectorParams | None = None,
      init_from_state: types.ProviderModelState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params,
        init_from_state=init_from_state)

    self._chosen_endpoint_cached_result = {}

    if init_from_state:
      if init_from_state.provider_model is None:
        raise ValueError('provider_model needs to be set in init_from_state.')
      if init_from_state.provider_model.provider != self.get_provider_name():
        raise ValueError(
            'provider_model needs to be same with the class provider name.\n'
            f'provider_model: {init_from_state.provider_model}\n'
            f'class provider name: {self.get_provider_name()}')
      self.load_state(init_from_state)
    else:
      self.provider_model = init_from_params.provider_model
      self.run_type = init_from_params.run_type
      self.provider_model_config = init_from_params.provider_model_config
      self.feature_mapping_strategy = init_from_params.feature_mapping_strategy
      self.query_cache_manager = init_from_params.query_cache_manager
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection

  def get_internal_state_property_name(self):
    return _PROVIDER_MODEL_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.ProviderModelState

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
      self,
      state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

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
        "Could not extract valid JSON from response",
        text,
        0)

  def _check_feature_exists(
      self,
      feature_name: types.FeatureNameType,
      query_record: types.QueryRecord) -> Any:
    if feature_name.startswith('response_format::'):
      response_format_type = feature_name.split('::')[1]
      return(
          query_record.response_format is not None and
          query_record.response_format.type == types.ResponseFormatType(
              response_format_type.upper()))
    else:
      return getattr(query_record, feature_name, None) is not None

  def _get_features_from_query_record(
      self,
      query_record: types.QueryRecord) -> types.FeatureListType:
    features = []
    for feature in types.FeatureNameType.__members__.values():
      if self._check_feature_exists(feature.value, query_record):
        features.append(feature)
    return features

  def _get_feature_signature(
      self,
      provider_model: types.ProviderModelType,
      feature_mapping_strategy: types.FeatureMappingStrategy,
      features: types.FeatureListType
  ) -> str:
    feature_signature = [
        str(provider_model),
        str(feature_mapping_strategy),
    ]
    for feature in features:
      feature_signature.append(feature.value)
    return ':::'.join(feature_signature)

  def _get_available_endpoints(
      self,
      features: types.FeatureListType) -> list[str]:
    supported_endpoints = []
    best_effort_endpoints = []
    for feature in features:
      supported_endpoints.append(set(
          self.provider_model_config.features[feature.value].supported))
      best_effort_endpoints.append(set(
          self.provider_model_config.features[feature.value].supported +
          self.provider_model_config.features[feature.value].best_effort))

    if supported_endpoints:
      supported_endpoints = list(
          reduce(set.intersection, supported_endpoints))

    if best_effort_endpoints:
      best_effort_endpoints = list(
          reduce(set.intersection, best_effort_endpoints))

    return supported_endpoints, best_effort_endpoints

  def _check_endpoints_usability(
      self,
      supported_endpoints: list[str],
      best_effort_endpoints: list[str],
      provider_model: types.ProviderModelType,
      feature_mapping_strategy: types.FeatureMappingStrategy,
      features: types.FeatureListType,
      raise_error: bool = True):

    if (feature_mapping_strategy ==
        types.FeatureMappingStrategy.STRICT):
      if len(supported_endpoints) == 0:
        if not raise_error:
          return False
        message = (
            f'For {provider_model}, it is not possible to ' +
            'use following features all at once in STRICT mode. ' +
            'Please consider to use BEST_EFFORT mode or remove some '
            'features.\n' +
            f'Requested features: {", ".join(features)}.')
        logging_utils.log_message(
            type=types.LoggingType.ERROR,
            logging_options=self.logging_options,
            message=message)
        raise Exception(message)
    elif (feature_mapping_strategy ==
          types.FeatureMappingStrategy.BEST_EFFORT):
      if (len(supported_endpoints) == 0 and
          len(best_effort_endpoints) == 0):
        if not raise_error:
          return False
        message = (
            f'For {provider_model}, it is not possible to ' +
            'use following features all at once in BEST_EFFORT mode. ' +
            'Please consider to remove some features.\n' +
            f'Requested features: {", ".join(features)}.')
        logging_utils.log_message(
            type=types.LoggingType.ERROR,
            logging_options=self.logging_options,
            message=message)
        raise Exception(message)
    return True

  def _select_endpoint(
      self,
      supported_endpoints: list[str],
      best_effort_endpoints: list[str]) -> str:
    if len(supported_endpoints) != 0:
      return sorted(supported_endpoints)[0]
    elif len(best_effort_endpoints) != 0:
      return sorted(best_effort_endpoints)[0]

  def _sanitize_system_feature(
      self,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint in self.provider_model_config.features[
        'system'].supported:
      return query_record
    query_record.prompt = f'{query_record.system}\n\n{query_record.prompt}'
    query_record.system = None
    return query_record

  def _sanitize_messages_feature(
      self,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint in self.provider_model_config.features[
        'messages'].supported:
      return query_record
    prompt = query_record.prompt
    result_prompt = '\n'.join(
        f'{message["role"].upper()}: {message["content"]}'
        for message in query_record.messages)
    if prompt is not None:
      result_prompt = f'{result_prompt}\n\nUSER: {prompt}'
    query_record.prompt = result_prompt
    query_record.messages = None
    return query_record

  def _sanitize_web_search_feature(
      self,
      query_record: types.QueryRecord):
    if not query_record.web_search:
      query_record.web_search = None
    return query_record

  def _omit_best_effort_feature(
      self,
      feature_name: types.FeatureNameType,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint not in self.provider_model_config.features[
          feature_name].supported:
        message = (
            f'{self.provider_model.model} does not support {feature_name} '
            'in BEST_EFFORT mode. Omitting this feature.'
            f'Model endpoint: {query_record.chosen_endpoint}')
        logging_utils.log_message(
            type=types.LoggingType.WARNING,
            logging_options=self.logging_options,
            query_record=query_record,
            message=message)
        setattr(query_record, feature_name, None)
    return query_record

  def _get_schema_guidance(self, query_record: types.QueryRecord) -> str:
    schema_guidance = ''
    if query_record.response_format.type == types.ResponseFormatType.JSON:
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
    return schema_guidance

  def _sanitize_response_format_feature(
      self,
      query_record: types.QueryRecord):
    schema_guidance = self._get_schema_guidance(query_record)

    if query_record.prompt is not None:
      query_record.prompt = (
          f'{query_record.prompt}\n\n{schema_guidance}')
    else:
      query_record.prompt = schema_guidance
    return query_record

  def _sanitize_query_record(
      self,
      query_record: types.QueryRecord):

    if (query_record.feature_mapping_strategy ==
        types.FeatureMappingStrategy.STRICT):
      return query_record

    if self._check_feature_exists(
          feature_name='messages', query_record=query_record):
      query_record = self._sanitize_messages_feature(
            query_record=query_record)

    if self._check_feature_exists(
          feature_name='system', query_record=query_record):
      query_record = self._sanitize_system_feature(
            query_record=query_record)

    for feature_name in ['stop', 'temperature', 'max_tokens']:
      if self._check_feature_exists(
          feature_name=feature_name, query_record=query_record):
        query_record = self._omit_best_effort_feature(
            feature_name=feature_name,
            query_record=query_record)

    if self._check_feature_exists(
          feature_name='web_search', query_record=query_record):
      query_record = self._sanitize_web_search_feature(
            query_record=query_record)

    for feature_name in [
        'response_format::json',
        'response_format::json_schema',
        'response_format::pydantic',
    ]:
      if self._check_feature_exists(
          feature_name=feature_name, query_record=query_record):
        query_record = self._sanitize_response_format_feature(
            query_record=query_record)

    return query_record

  def _get_feature_check_result_endpoint(
      self,
      provider_model: types.ProviderModelType,
      feature_mapping_strategy: types.FeatureMappingStrategy,
      features: types.FeatureListType,
      raise_error: bool = True) -> str:
    feature_signature = self._get_feature_signature(
        provider_model=provider_model,
        feature_mapping_strategy=feature_mapping_strategy,
        features=features)

    if (self._chosen_endpoint_cached_result is not None and
        feature_signature in self._chosen_endpoint_cached_result):
      chosen_endpoint = self._chosen_endpoint_cached_result[feature_signature]
    else:
      supported_endpoints, best_effort_endpoints = self._get_available_endpoints(
          features=features)

      is_endpoints_usable = self._check_endpoints_usability(
          supported_endpoints=supported_endpoints,
          best_effort_endpoints=best_effort_endpoints,
          provider_model=provider_model,
          feature_mapping_strategy=feature_mapping_strategy,
          features=features,
          raise_error=raise_error)

      if not is_endpoints_usable:
        # Note: This set to False to indicate that the endpoints are not usable.
        # Cache is checking if the value is None or not. If value set to False,
        # cache will return False instead of retrying.
        chosen_endpoint = False
      else:
        self._chosen_endpoint_cached_result[
            feature_signature] = self._select_endpoint(
                supported_endpoints=supported_endpoints,
                best_effort_endpoints=best_effort_endpoints)

        chosen_endpoint = self._chosen_endpoint_cached_result[
            feature_signature]

    return chosen_endpoint

  def feature_check_and_sanitize(
      self,
      query_record: types.QueryRecord) -> types.QueryRecord:
    query_record = copy.deepcopy(query_record)

    chosen_endpoint = self._get_feature_check_result_endpoint(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=self._get_features_from_query_record(query_record),
        raise_error=True)

    query_record.chosen_endpoint = chosen_endpoint

    query_record = self._sanitize_query_record(query_record=query_record)

    return query_record

  def check_feature_compatibility(
      self,
      features: types.FeatureListType) -> bool:
    chosen_endpoint = self._get_feature_check_result_endpoint(
        provider_model=self.provider_model,
        feature_mapping_strategy=self.feature_mapping_strategy,
        features=features,
        raise_error=False)

    return bool(chosen_endpoint)

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

  def _temp_response_format_text_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    # Almost always, nothing done here. Because of that, this function does
    # not delegated to inherited classes.
    return query_function

  def add_features_to_query_function(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    feature_mappings = {
        types.FeatureNameType.PROMPT: self.prompt_feature_mapping,
        types.FeatureNameType.MESSAGES: self.messages_feature_mapping,
        types.FeatureNameType.SYSTEM: self.system_feature_mapping,
        types.FeatureNameType.MAX_TOKENS: self.max_tokens_feature_mapping,
        types.FeatureNameType.TEMPERATURE: self.temperature_feature_mapping,
        types.FeatureNameType.STOP: self.stop_feature_mapping,
        types.FeatureNameType.WEB_SEARCH: self.web_search_feature_mapping,
    }
    for feature_name in feature_mappings:
      if not self._check_feature_exists(
          feature_name=feature_name.value,
          query_record=query_record):
        continue
      query_function = feature_mappings[feature_name](
          query_record=query_record,
          query_function=query_function)

    response_format_feature_mappings = {
        types.FeatureNameType.RESPONSE_FORMAT_TEXT: (
            self._temp_response_format_text_feature_mapping),
        types.FeatureNameType.RESPONSE_FORMAT_JSON: self.json_feature_mapping,
        types.FeatureNameType.RESPONSE_FORMAT_JSON_SCHEMA: (
            self.json_schema_feature_mapping),
        types.FeatureNameType.RESPONSE_FORMAT_PYDANTIC: (
            self.pydantic_feature_mapping),
    }
    response_format = query_record.response_format
    if response_format is not None:
      response_format_str = (
          f'response_format::{response_format.type.value.lower()}')
      if query_record.chosen_endpoint in self.provider_model_config.features[
            response_format_str].supported:
        query_function = response_format_feature_mappings[response_format_str](
            query_record=query_record,
            query_function=query_function)
    return query_function

  def _handle_json_response_format(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if (query_record.chosen_endpoint in
        self.provider_model_config.features['response_format::json'].supported):
      return types.Response(
          value=self.format_json_response_from_provider(
              response=response,
              query_record=query_record),
          type=types.ResponseType.JSON)
    else:
      return types.Response(
          value=self._extract_json_from_text(
              self.format_text_response_from_provider(
                  response=response,
                  query_record=query_record)),
          type=types.ResponseType.JSON)

  def _handle_json_schema_response_format(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if (query_record.chosen_endpoint in
        self.provider_model_config.features[
            'response_format::json_schema'].supported):
      return types.Response(
          value=self.format_json_schema_response_from_provider(
              response=response,
              query_record=query_record),
          type=types.ResponseType.JSON)
    else:
      return types.Response(
          value=self._extract_json_from_text(
              self.format_text_response_from_provider(
                  response=response,
                  query_record=query_record)),
          type=types.ResponseType.JSON)

  def _handle_pydantic_response_format(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if (query_record.chosen_endpoint in
        self.provider_model_config.features[
            'response_format::pydantic'].supported):
      instance = self.format_pydantic_response_from_provider(
          response=response,
          query_record=query_record)
      return types.Response(
          value=instance,
          type=types.ResponseType.PYDANTIC,
          pydantic_metadata=types.PydanticMetadataType(
              class_name=query_record.response_format.value.class_name))
    else:
      json_value = self._extract_json_from_text(
          self.format_text_response_from_provider(
              response=response,
              query_record=query_record))
      pydantic_class = query_record.response_format.value.class_value
      instance = pydantic_class.model_validate(json_value)
      return types.Response(
          value=instance,
          type=types.ResponseType.PYDANTIC,
          pydantic_metadata=types.PydanticMetadataType(
              class_name=query_record.response_format.value.class_name))

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
      return self._handle_json_response_format(
          response=response,
          query_record=query_record)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return self._handle_json_schema_response_format(
          response=response,
          query_record=query_record)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return self._handle_pydantic_response_format(
          response=response,
          query_record=query_record)

  def get_token_count_estimate(
      self,
      value: str | types.Response | types.MessagesType | None = None) -> int:
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
        # Try pydantic_metadata.instance_json_value first, then value (instance)
        if (value.pydantic_metadata is not None and
            value.pydantic_metadata.instance_json_value is not None):
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.pydantic_metadata.instance_json_value))
        elif value.value is not None:
          total += _get_token_count_estimate_from_prompt(
              json.dumps(value.value.model_dump()))
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
        response_token_count * response_token_cost)

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
      prompt: str | None = None,
      system: str | None = None,
      messages: types.MessagesType | None = None,
      max_tokens: int | None = None,
      temperature: float | None = None,
      stop: types.StopType | None = None,
      response_format: types.ResponseFormat | None = None,
      web_search: bool | None = None,
      provider_model: types.ProviderModelIdentifierType | None = None,
      feature_mapping_strategy: types.FeatureMappingStrategy | None = None,
      use_cache: bool = True,
      unique_response_limit: int | None = None) -> types.LoggingRecord:
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages is not None:
      type_utils.check_messages_type(messages)

    if response_format is None:
      response_format = types.ResponseFormat(
          type=types.ResponseFormatType.TEXT)

    if provider_model is not None:
      if isinstance(provider_model, tuple):
        provider_model = self.provider_model_config.provider_model
      if provider_model != self.provider_model:
        raise ValueError(
            'provider_model does not match the connector provider_model.'
            f'provider_model: {provider_model}\n'
            f'connector provider_model: {self.provider_model}')

    if feature_mapping_strategy is None:
      feature_mapping_strategy = self.feature_mapping_strategy

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
        feature_mapping_strategy=feature_mapping_strategy,
        token_count=self.get_token_count_estimate(
            value = prompt if prompt is not None else messages))

    query_record = self.feature_check_and_sanitize(query_record=query_record)

    look_fail_reason = None
    if self.query_cache_manager and use_cache:
      cache_look_result = None
      response_record = None
      try:
        cache_look_result = self.query_cache_manager.look(
            query_record,
            unique_response_limit=unique_response_limit)
        if cache_look_result.query_response:
          response_record = cache_look_result.query_response
      except Exception:
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
      response = self.generate_text_proc(query_record=query_record)
    except Exception as e:
      error_traceback = traceback.format_exc()
      error = e

    if response is not None:
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
          query_record=query_record,
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
    self._update_proxdash(logging_record=logging_record)
    return logging_record

  def get_provider_name(self):
    raise NotImplementedError

  def init_model(self):
    raise NotImplementedError

  def init_mock_model(self):
    raise NotImplementedError

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    raise NotImplementedError

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    raise NotImplementedError

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    raise NotImplementedError

  def temperature_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    raise NotImplementedError

  def stop_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
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

  def web_search_feature_mapping(
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

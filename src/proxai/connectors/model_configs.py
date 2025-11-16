from __future__ import annotations

import json
from typing import List
from importlib import resources
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller

_MODEL_CONFIGS_STATE_PROPERTY = '_model_configs_state'

PROVIDER_KEY_MAP: Dict[str, Tuple[str]] = MappingProxyType({
    'claude': tuple(['ANTHROPIC_API_KEY']),
    'cohere': tuple(['CO_API_KEY']),
    'databricks': tuple(['DATABRICKS_TOKEN', 'DATABRICKS_HOST']),
    'deepseek': tuple(['DEEPSEEK_API_KEY']),
    'gemini': tuple(['GEMINI_API_KEY']),
    'grok': tuple(['XAI_API_KEY']),
    'huggingface': tuple(['HUGGINGFACE_API_KEY']),
    'mistral': tuple(['MISTRAL_API_KEY']),
    'openai': tuple(['OPENAI_API_KEY']),

    'mock_provider': tuple(['MOCK_PROVIDER_API_KEY']),
    'mock_failing_provider': tuple(['MOCK_FAILING_PROVIDER']),
    'mock_slow_provider': tuple(['MOCK_SLOW_PROVIDER']),
})


class ModelConfigs(state_controller.StateControlled):
  _model_configs_schema: Optional[types.ModelConfigsSchemaType]
  _model_configs_state: Optional[types.ModelConfigsState]

  LOCAL_CONFIG_VERSION = "v1.0.0"

  def __init__(
      self,
      model_configs_schema: Optional[types.ModelConfigsSchemaType] = None,
      init_state=None):
    super().__init__(
        model_configs_schema=model_configs_schema,
        init_state=init_state)

    if init_state:
      self.load_state(init_state)
    else:
      initial_state = self.get_state()

      if model_configs_schema is None:
        model_configs_schema = self._load_config_from_local_json_file()
      self.model_configs_schema = model_configs_schema
      self.handle_changes(initial_state, self.get_state())

  def get_internal_state_property_name(self):
    return _MODEL_CONFIGS_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.ModelConfigsState

  def handle_changes(
      self,
      old_state: types.ModelConfigsState,
      current_state: types.ModelConfigsState):
    pass

  @property
  def model_configs_schema(self) -> types.ModelConfigsSchemaType:
    return self.get_property_value('model_configs_schema')

  @model_configs_schema.setter
  def model_configs_schema(self, value: types.ModelConfigsSchemaType):
    internal_value = self.get_property_internal_state_value(
        'model_configs_schema')
    if internal_value != value:
      self._validate_model_configs_schema(value)
    self.set_property_value('model_configs_schema', value)

  def _validate_model_configs_schema(
      self,
      model_configs_schema: types.ModelConfigsSchemaType):
    # TODO: Partially finished. Need to add more validation and tests.
    provider_model_configs = model_configs_schema.version_config.provider_model_configs
    featured_models = model_configs_schema.version_config.featured_models
    models_by_call_type = model_configs_schema.version_config.models_by_call_type
    models_by_size = model_configs_schema.version_config.models_by_size
    for provider, model_configs in provider_model_configs.items():
      for provider_model_identifier, model_config in model_configs.items():
        self.check_provider_model_identifier_type(
            model_config.provider_model,
            model_configs_schema)

    for provider, provider_models in featured_models.items():
      for provider_model_identifier in provider_models:
        self.check_provider_model_identifier_type(
            provider_model_identifier,
            model_configs_schema)

    for call_type, provider_models in models_by_call_type.items():
      for provider, provider_models in provider_models.items():
        for provider_model_identifier in provider_models:
          self.check_provider_model_identifier_type(
              provider_model_identifier,
              model_configs_schema)

    for model_size, provider_models in models_by_size.items():
      for provider_model_identifier in provider_models:
        self.check_provider_model_identifier_type(
            provider_model_identifier,
            model_configs_schema)

  def _is_provider_model_tuple(self, value: Any) -> bool:
    return (
        type(value) == tuple
        and len(value) == 2
        and type(value[0]) == str
        and type(value[1]) == str)

  @staticmethod
  def _load_config_from_local_json_file(
      version: Optional[str] = None) -> types.ModelConfigsSchemaType:
    version = version or ModelConfigs.LOCAL_CONFIG_VERSION

    try:
      config_data = (
          resources.files("proxai.connectors.model_configs_data")
          .joinpath(f"{version}.json")
          .read_text(encoding="utf-8")
      )
    except FileNotFoundError:
      raise FileNotFoundError(
          f'Model config file "{version}.json" not found in package. '
          'Please update the proxai package to the latest version. '
          'If updating does not resolve the issue, please contact support@proxai.co'
      )

    try:
      config_dict = json.loads(config_data)
    except json.JSONDecodeError as e:
      raise ValueError(
        f'Invalid JSON in config file "{version}.json". '
        'Please update the proxai package to the latest version. '
        'If updating does not resolve the issue, please contact support@proxai.co\n'
        f'Error: {e}')

    return type_serializer.decode_model_configs_schema_type(config_dict)

  def check_provider_model_identifier_type(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType,
      model_configs_schema: Optional[types.ModelConfigsSchemaType] = None):
    """Check if provider model identifier is supported."""
    if model_configs_schema is None:
      model_configs_schema = self.model_configs_schema
    provider_model_configs = model_configs_schema.version_config.provider_model_configs
    if isinstance(provider_model_identifier, types.ProviderModelType):
      provider = provider_model_identifier.provider
      model = provider_model_identifier.model
      if provider not in provider_model_configs:
        raise ValueError(
          f'Provider not supported: {provider}.\n'
          f'Supported providers: {list(provider_model_configs.keys())}')
      if model not in provider_model_configs[provider]:
        raise ValueError(
          f'Model not supported: {model}.\nSupported models: '
          f'{provider_model_configs[provider].keys()}')
      config_provider_model = (
          provider_model_configs[provider][model].provider_model)
      if provider_model_identifier != config_provider_model:
        raise ValueError(
          'Mismatch between provider model identifier and model config.'
          f'Provider model identifier: {provider_model_identifier}'
          f'Model config: {config_provider_model}')
    elif self._is_provider_model_tuple(provider_model_identifier):
      provider = provider_model_identifier[0]
      model = provider_model_identifier[1]
      if provider not in provider_model_configs:
        raise ValueError(
          f'Provider not supported: {provider}.\n'
          f'Supported providers: {provider_model_configs.keys()}')
      if model not in provider_model_configs[provider]:
        raise ValueError(
          f'Model not supported: {model}.\nSupported models: '
          f'{provider_model_configs[provider].keys()}')
    else:
      raise ValueError(
          f'Invalid provider model identifier: {provider_model_identifier}')

  def get_provider_model(
      self,
      model_identifier: types.ProviderModelIdentifierType
  ) -> types.ProviderModelType:
    if self._is_provider_model_tuple(model_identifier):
      return self.model_configs_schema.provider_model_configs[
          model_identifier[0]][model_identifier[1]].provider_model
    else:
      return model_identifier

  def get_provider_model_config(
      self,
      model_identifier: types.ProviderModelIdentifierType
  ) -> types.ProviderModelType:
    provider_model = self.get_provider_model(model_identifier)
    return self.model_configs_schema.provider_model_configs[
        provider_model.provider][provider_model.model]

  def get_provider_model_cost(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType,
      query_token_count: int,
      response_token_count: int,
  ) -> int:
    provider_model = self.get_provider_model_config(
        provider_model_identifier)
    model_pricing_config = self.model_configs_schema.provider_model_configs[
      provider_model.provider][provider_model.model].pricing
    return math.floor(
        query_token_count * model_pricing_config.per_query_token_cost +
        response_token_count * model_pricing_config.per_response_token_cost)

  def is_feature_supported(
      self,
      provider_model: types.ProviderModelType,
      feature: str,
  ) -> bool:
    model_features = self.model_configs_schema.provider_model_configs[
        provider_model.provider][provider_model.model].features
    if model_features is None:
      return True
    return feature in model_features.not_supported_features

  def get_all_models(
      self,
      provider: Optional[types.ProviderNameType] = None,
      model_size: Optional[types.ModelSizeType] = None,
      call_type: Optional[types.CallType] = types.CallType.GENERATE_TEXT,
      only_featured: Optional[bool] = True,
  ) -> List[types.ProviderModelType]:
    if (call_type is not None and
        call_type not in self.model_configs_schema.models_by_call_type):
      raise ValueError(f'Call type not supported: {call_type}')

    if (model_size is not None and
        model_size not in self.model_configs_schema.models_by_size):
      raise ValueError(f'Model size not supported: {model_size}')

    if (provider is not None and
        provider not in self.model_configs_schema.provider_model_configs):
      supported_providers = list(
          self.model_configs_schema.provider_model_configs.keys())
      raise ValueError(
          f'Provider not supported: {provider}.\n'
          f'Supported providers: {supported_providers}')

    result_provider_models = []
    for provider_name, provider_models in (
        self.model_configs_schema.provider_model_configs.items()):
      if provider is not None and provider_name != provider:
          continue

      for provider_model_config in provider_models.values():
        if (call_type is not None and
            provider_model_config.metadata.call_type != call_type):
          continue

        if (model_size is not None and
            provider_model_config.metadata.model_size != model_size):
          continue

        if only_featured and not provider_model_config.metadata.is_featured:
          continue

        result_provider_models.append(provider_model_config.provider_model)

    return result_provider_models

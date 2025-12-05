from __future__ import annotations

import json
import math
from typing import List, Set
from importlib import resources
from importlib.metadata import version
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging.version import Version, InvalidVersion
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
        model_configs_schema = self._load_model_config_schema_from_local_files()
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

  def _validate_min_proxai_version(
      self,
      min_proxai_version: Optional[str]):
    if min_proxai_version is None:
      return

    current_version = version("proxai")

    try:
      specifier_set = SpecifierSet(min_proxai_version)
      current = Version(current_version)

      if not specifier_set.contains(current):
        raise ValueError(
            f'Current proxai version ({current_version}) does not satisfy the minimum '
            f'version requirement: {min_proxai_version}. '
            f'Please upgrade proxai to a version that satisfies this requirement.')
    except InvalidSpecifier as e:
      raise ValueError(
          f'Model configs schema metadata min_proxai_version is invalid. '
          f'Min proxai version specifier: {min_proxai_version}. '
          f'Error: {e}')
    except InvalidVersion as e:
      raise ValueError(
          f'Current proxai version ({current_version}) is invalid. '
          f'Error: {e}')

  def _validate_model_configs_schema_metadata(
      self,
      model_configs_schema_metadata: types.ModelConfigsSchemaMetadataType):
    self._validate_min_proxai_version(
        model_configs_schema_metadata.min_proxai_version)

  def _get_provider_model_key(
      self,
      provider_model: types.ProviderModelIdentifierType
  ) -> Tuple[str, str]:
    """Extract (provider, model) tuple from any provider model identifier."""
    if isinstance(provider_model, types.ProviderModelType):
      return (provider_model.provider, provider_model.model)
    elif self._is_provider_model_tuple(provider_model):
      return (provider_model[0], provider_model[1])
    raise ValueError(f'Invalid provider model identifier: {provider_model}')

  def _get_all_featured_models_from_configs(
      self,
      provider_model_configs: types.ProviderModelConfigsType
  ) -> Set[Tuple[str, str]]:
    """Get set of (provider, model) tuples for all featured models in configs."""
    featured = set()
    for provider, models in provider_model_configs.items():
      for model_name, config in models.items():
        if config.metadata and config.metadata.is_featured:
          featured.add((provider, model_name))
    return featured

  def _get_all_models_by_call_type_from_configs(
      self,
      provider_model_configs: types.ProviderModelConfigsType
  ) -> Dict[types.CallType, Set[Tuple[str, str]]]:
    """Get models grouped by call_type from configs."""
    by_call_type: Dict[types.CallType, Set[Tuple[str, str]]] = {}
    for provider, models in provider_model_configs.items():
      for model_name, config in models.items():
        if config.metadata and config.metadata.call_type:
          call_type = config.metadata.call_type
          if call_type not in by_call_type:
            by_call_type[call_type] = set()
          by_call_type[call_type].add((provider, model_name))
    return by_call_type

  def _get_all_models_by_size_from_configs(
      self,
      provider_model_configs: types.ProviderModelConfigsType
  ) -> Dict[types.ModelSizeType, Set[Tuple[str, str]]]:
    """Get models grouped by size from configs."""
    by_size: Dict[types.ModelSizeType, Set[Tuple[str, str]]] = {}
    for provider, models in provider_model_configs.items():
      for model_name, config in models.items():
        if config.metadata and config.metadata.model_size_tags:
          for size_tag in config.metadata.model_size_tags:
            if size_tag not in by_size:
              by_size[size_tag] = set()
            by_size[size_tag].add((provider, model_name))
    return by_size

  def _validate_provider_model_key_matches_config(
      self,
      provider_key: str,
      model_key: str,
      config: types.ProviderModelConfigType):
    """Validate provider_model fields match the dict keys."""
    if config.provider_model is None:
      raise ValueError(
          f'provider_model is None for config at '
          f'provider_model_configs[{provider_key}][{model_key}]')

    if config.provider_model.provider != provider_key:
      raise ValueError(
          f'Provider mismatch: config key is "{provider_key}" but '
          f'provider_model.provider is "{config.provider_model.provider}"')

    if config.provider_model.model != model_key:
      raise ValueError(
          f'Model mismatch: config key is "{model_key}" but '
          f'provider_model.model is "{config.provider_model.model}"')

  def _validate_pricing(
      self,
      provider_key: str,
      model_key: str,
      pricing: types.ProviderModelPricingType):
    """Validate pricing values are non-negative."""
    if pricing is None:
      raise ValueError(
          f'pricing is None for provider_model_configs[{provider_key}][{model_key}]')

    if pricing.per_query_token_cost < 0:
      raise ValueError(
          f'per_query_token_cost is negative ({pricing.per_query_token_cost}) '
          f'for provider_model_configs[{provider_key}][{model_key}]')

    if pricing.per_response_token_cost < 0:
      raise ValueError(
          f'per_response_token_cost is negative ({pricing.per_response_token_cost}) '
          f'for provider_model_configs[{provider_key}][{model_key}]')

  def _validate_model_size_tags(
      self,
      provider_key: str,
      model_key: str,
      model_size_tags: List[types.ModelSizeType]):
    """Validate model_size_tags contains only valid ModelSizeType values."""
    valid_sizes = {size for size in types.ModelSizeType}
    for tag in model_size_tags:
      if tag not in valid_sizes:
        raise ValueError(
            f'Invalid model_size_tag "{tag}" for '
            f'provider_model_configs[{provider_key}][{model_key}]. '
            f'Valid values: {[s.value for s in types.ModelSizeType]}')

  def _validate_provider_model_config(
      self,
      provider_key: str,
      model_key: str,
      config: types.ProviderModelConfigType):
    """Validate a single ProviderModelConfigType."""
    self._validate_provider_model_key_matches_config(
        provider_key, model_key, config)

    self._validate_pricing(provider_key, model_key, config.pricing)

    if (config.metadata and
        config.metadata.model_size_tags is not None):
      self._validate_model_size_tags(
          provider_key, model_key, config.metadata.model_size_tags)

  def _validate_provider_model_configs(
      self,
      provider_model_configs: types.ProviderModelConfigsType):
    """Validate all provider model configs."""
    for provider_key, models in provider_model_configs.items():
      for model_key, config in models.items():
        self._validate_provider_model_config(provider_key, model_key, config)

  def _validate_featured_models(
      self,
      provider_model_configs: types.ProviderModelConfigsType,
      featured_models: types.FeaturedModelsType):
    """Validate featured_models matches is_featured in provider_model_configs."""
    featured_from_configs = self._get_all_featured_models_from_configs(
        provider_model_configs)

    featured_from_list: Set[Tuple[str, str]] = set()
    for provider, models in featured_models.items():
      for model in models:
        key = self._get_provider_model_key(model)
        featured_from_list.add(key)

    missing_in_list = featured_from_configs - featured_from_list
    if missing_in_list:
      raise ValueError(
          f'Models marked as is_featured=True in provider_model_configs '
          f'but missing from featured_models: {sorted(missing_in_list)}')

    extra_in_list = featured_from_list - featured_from_configs
    if extra_in_list:
      raise ValueError(
          f'Models in featured_models but not marked as is_featured=True '
          f'in provider_model_configs: {sorted(extra_in_list)}')

  def _validate_models_by_call_type(
      self,
      provider_model_configs: types.ProviderModelConfigsType,
      models_by_call_type: types.ModelsByCallTypeType):
    """Validate models_by_call_type matches call_type in provider_model_configs."""
    from_configs = self._get_all_models_by_call_type_from_configs(
        provider_model_configs)

    from_list: Dict[types.CallType, Set[Tuple[str, str]]] = {}
    for call_type, providers in models_by_call_type.items():
      from_list[call_type] = set()
      for provider, models in providers.items():
        for model in models:
          key = self._get_provider_model_key(model)
          from_list[call_type].add(key)

    all_call_types = set(from_configs.keys()) | set(from_list.keys())
    for call_type in all_call_types:
      config_models = from_configs.get(call_type, set())
      list_models = from_list.get(call_type, set())

      missing_in_list = config_models - list_models
      if missing_in_list:
        raise ValueError(
            f'Models with call_type={call_type} in provider_model_configs '
            f'but missing from models_by_call_type: {sorted(missing_in_list)}')

      extra_in_list = list_models - config_models
      if extra_in_list:
        raise ValueError(
            f'Models in models_by_call_type[{call_type}] but not marked with '
            f'that call_type in provider_model_configs: {sorted(extra_in_list)}')

  def _validate_models_by_size(
      self,
      provider_model_configs: types.ProviderModelConfigsType,
      models_by_size: types.ModelsBySizeType):
    """Validate models_by_size matches model_size_tags in provider_model_configs."""
    from_configs = self._get_all_models_by_size_from_configs(
        provider_model_configs)

    from_list: Dict[types.ModelSizeType, Set[Tuple[str, str]]] = {}
    for size, models in models_by_size.items():
      from_list[size] = set()
      for model in models:
        key = self._get_provider_model_key(model)
        from_list[size].add(key)

    all_sizes = set(from_configs.keys()) | set(from_list.keys())
    for size in all_sizes:
      config_models = from_configs.get(size, set())
      list_models = from_list.get(size, set())

      missing_in_list = config_models - list_models
      if missing_in_list:
        raise ValueError(
            f'Models with model_size_tags containing {size} in '
            f'provider_model_configs but missing from models_by_size: '
            f'{sorted(missing_in_list)}')

      extra_in_list = list_models - config_models
      if extra_in_list:
        raise ValueError(
            f'Models in models_by_size[{size}] but model_size_tags in '
            f'provider_model_configs does not contain {size}: '
            f'{sorted(extra_in_list)}')

  def _validate_default_model_priority_list(
      self,
      provider_model_configs: types.ProviderModelConfigsType,
      default_model_priority_list: types.DefaultModelPriorityListType):
    """Validate all models in default_model_priority_list exist in configs."""
    for model in default_model_priority_list:
      key = self._get_provider_model_key(model)
      provider, model_name = key
      if provider not in provider_model_configs:
        raise ValueError(
            f'Provider {provider} in default_model_priority_list '
            f'not found in provider_model_configs')
      if model_name not in provider_model_configs[provider]:
        raise ValueError(
            f'Model {model_name} for provider {provider} in '
            f'default_model_priority_list not found in provider_model_configs')

  def _validate_version_config(
      self,
      version_config: types.ModelConfigsSchemaVersionConfigType):
    """Validate version_config internal consistency."""
    provider_model_configs = version_config.provider_model_configs
    if provider_model_configs is None:
      return

    self._validate_provider_model_configs(provider_model_configs)

    if version_config.featured_models is not None:
      self._validate_featured_models(
          provider_model_configs, version_config.featured_models)

    if version_config.models_by_call_type is not None:
      self._validate_models_by_call_type(
          provider_model_configs, version_config.models_by_call_type)

    if version_config.models_by_size is not None:
      self._validate_models_by_size(
          provider_model_configs, version_config.models_by_size)

    if version_config.default_model_priority_list is not None:
      self._validate_default_model_priority_list(
          provider_model_configs, version_config.default_model_priority_list)

  def _validate_model_configs_schema(
      self,
      model_configs_schema: types.ModelConfigsSchemaType):
    if model_configs_schema.metadata:
      self._validate_model_configs_schema_metadata(model_configs_schema.metadata)

    if model_configs_schema.version_config:
      self._validate_version_config(model_configs_schema.version_config)

  def _is_provider_model_tuple(self, value: Any) -> bool:
    return (
        type(value) == tuple
        and len(value) == 2
        and type(value[0]) == str
        and type(value[1]) == str)

  @staticmethod
  def _load_model_config_schema_from_local_files(
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

  def load_model_config_from_json_string(
      self,
      json_string: str):
    model_configs_schema = type_serializer.decode_model_configs_schema_type(
        json.loads(json_string))
    self.model_configs_schema = model_configs_schema

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
      return self.model_configs_schema.version_config.provider_model_configs[
          model_identifier[0]][model_identifier[1]].provider_model
    else:
      return model_identifier

  def get_provider_model_config(
      self,
      model_identifier: types.ProviderModelIdentifierType
  ) -> types.ProviderModelType:
    provider_model = self.get_provider_model(model_identifier)
    return self.model_configs_schema.version_config.provider_model_configs[
        provider_model.provider][provider_model.model]

  def get_provider_model_cost(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType,
      query_token_count: int,
      response_token_count: int,
  ) -> int:
    provider_model = self.get_provider_model(
        provider_model_identifier)
    version_config = self.model_configs_schema.version_config
    model_pricing_config = version_config.provider_model_configs[
      provider_model.provider][provider_model.model].pricing
    return math.floor(
        query_token_count * model_pricing_config.per_query_token_cost +
        response_token_count * model_pricing_config.per_response_token_cost)

  def is_feature_supported(
      self,
      provider_model: types.ProviderModelType,
      feature: str,
  ) -> bool:
    version_config = self.model_configs_schema.version_config
    model_features = version_config.provider_model_configs[
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
    version_config = self.model_configs_schema.version_config
    if (call_type is not None and
        call_type not in version_config.models_by_call_type):
      raise ValueError(f'Call type not supported: {call_type}')

    if (model_size is not None and
        model_size not in version_config.models_by_size):
      raise ValueError(f'Model size not supported: {model_size}')

    if (provider is not None and
        provider not in version_config.provider_model_configs):
      supported_providers = list(
          version_config.provider_model_configs.keys())
      raise ValueError(
          f'Provider not supported: {provider}.\n'
          f'Supported providers: {supported_providers}')

    result_provider_models = []
    for provider_name, provider_models in (
        version_config.provider_model_configs.items()):
      if provider is not None and provider_name != provider:
          continue

      for provider_model_config in provider_models.values():
        if (call_type is not None and
            provider_model_config.metadata.call_type != call_type):
          continue

        if (model_size is not None and (
            provider_model_config.metadata.model_size_tags is None or
            model_size not in provider_model_config.metadata.model_size_tags)):
          continue

        if only_featured and not provider_model_config.metadata.is_featured:
          continue

        result_provider_models.append(provider_model_config.provider_model)

    return result_provider_models

  def get_default_model_priority_list(self) -> List[types.ProviderModelType]:
    # TODO: This operation could be optimized by caching the result and using
    # StateController to persist the result. If the configs are updated, the
    # result should be invalidated and recalculated by the StateController's
    # handle_changes method.
    result = []
    for provider_model in self.model_configs_schema.version_config.default_model_priority_list:
      result.append(self.get_provider_model(provider_model))
    return result

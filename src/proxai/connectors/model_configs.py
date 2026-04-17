from __future__ import annotations

import dataclasses
import json
import math
from importlib import resources
from importlib.metadata import version
from types import MappingProxyType
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_MODEL_CONFIGS_STATE_PROPERTY = '_model_configs_state'

PROVIDER_KEY_MAP: dict[str, tuple[str]] = MappingProxyType({
    'claude': ('ANTHROPIC_API_KEY',),
    'cohere': ('CO_API_KEY',),
    'databricks': ('DATABRICKS_TOKEN', 'DATABRICKS_HOST'),
    'deepseek': ('DEEPSEEK_API_KEY',),
    'gemini': ('GEMINI_API_KEY',),
    'grok': ('XAI_API_KEY',),
    'huggingface': ('HF_TOKEN',),
    'mistral': ('MISTRAL_API_KEY',),
    'openai': ('OPENAI_API_KEY',),
    'mock_provider': ('MOCK_PROVIDER_API_KEY',),
    'mock_failing_provider': ('MOCK_FAILING_PROVIDER',),
    'mock_slow_provider': ('MOCK_SLOW_PROVIDER',),
})


@dataclasses.dataclass
class ModelConfigsParams:
  """Initialization parameters for ModelConfigs."""

  model_registry: types.ModelRegistry | None = None


class ModelConfigs(state_controller.StateControlled):
  """Manages model configuration schemas and validation."""

  _model_registry: types.ModelRegistry | None
  _models_by_model_size: types.ModelSizeMappingType | None
  _recommended_models: types.RecommendedModelsMappingType | None
  _model_configs_state: types.ModelConfigsState | None

  LOCAL_CONFIG_VERSION = "v1.2.0"

  def __init__(  # noqa: D107
      self,
      init_from_params: ModelConfigsParams | None = None,
      init_from_state: types.ModelConfigsState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    if init_from_state:
      self.load_state(init_from_state)
    else:
      if not init_from_params or init_from_params.model_registry is None:
        model_registry = self._load_model_registry_from_local_files()
      else:
        model_registry = init_from_params.model_registry
      self.model_registry = types.ModelRegistry(
          metadata=model_registry.metadata,
          default_model_priority_list=(
              model_registry.default_model_priority_list),
          provider_model_configs={},
      )
      for models in model_registry.provider_model_configs.values():
        for model_config in models.values():
          self.register_provider_model_config(model_config)

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _MODEL_CONFIGS_STATE_PROPERTY

  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    return types.ModelConfigsState

  @property
  def model_registry(self) -> types.ModelRegistry:
    return self.get_property_value('model_registry')

  @model_registry.setter
  def model_registry(self, value: types.ModelRegistry):
    self.set_property_value('model_registry', value)

  @property
  def models_by_model_size(self) -> types.ModelSizeMappingType:
    return self.get_property_value('models_by_model_size')

  @models_by_model_size.setter
  def models_by_model_size(self, value: types.ModelSizeMappingType):
    self.set_property_value('models_by_model_size', value)

  @property
  def recommended_models(self) -> types.RecommendedModelsMappingType:
    return self.get_property_value('recommended_models')

  @recommended_models.setter
  def recommended_models(self, value: types.RecommendedModelsMappingType):
    self.set_property_value('recommended_models', value)

  def register_provider_model_config(
      self,
      provider_model_config: types.ProviderModelConfig):
    provider = provider_model_config.provider_model.provider
    model = provider_model_config.provider_model.model
    if provider not in self.model_registry.provider_model_configs:
      self.model_registry.provider_model_configs[provider] = {}
    if model in self.model_registry.provider_model_configs[provider]:
      raise ValueError(
          f'Model {model} already registered for provider {provider}. '
          'Please use a different model or delete the existing model.'
      )
    self.model_registry.provider_model_configs[
        provider][model] = provider_model_config

    if not self.models_by_model_size:
        self.models_by_model_size = {}
    if provider_model_config.metadata.model_size_tags:
      for model_size_tag in provider_model_config.metadata.model_size_tags:
        if model_size_tag not in self.models_by_model_size:
          self.models_by_model_size[model_size_tag] = []
        self.models_by_model_size[model_size_tag].append(
            provider_model_config.provider_model
        )

    if not self.recommended_models:
      self.recommended_models = {}
    if provider_model_config.metadata.is_recommended:
      if provider not in self.recommended_models:
        self.recommended_models[provider] = []
      self.recommended_models[provider].append(
          provider_model_config.provider_model
      )

  def unregister_model(
      self,
      provider_model: types.ProviderModelType):
    provider = provider_model.provider
    model = provider_model.model
    if provider not in self.model_registry.provider_model_configs:
      raise ValueError(
          f'Provider {provider} not registered.'
      )
    if model not in self.model_registry.provider_model_configs[provider]:
      raise ValueError(
          f'Model {model} not registered for provider {provider}.'
      )
    
    provider_model_config = self.model_registry.provider_model_configs[
        provider][model]
    if (provider_model_config.provider_model.provider_model_identifier !=
        provider_model.provider_model_identifier):
      raise ValueError(
          'Provider model identifier mismatch: '
          f'{provider_model_config.provider_model.provider_model_identifier} '
          f'!= {provider_model.provider_model_identifier}')
    
    for model_size_tag in provider_model_config.metadata.model_size_tags:
      if model_size_tag in self.models_by_model_size:
        self.models_by_model_size[model_size_tag].remove(
            provider_model
        )
    if provider_model_config.metadata.is_recommended:
      if provider in self.recommended_models:
        self.recommended_models[provider].remove(
            provider_model
        )

    del self.model_registry.provider_model_configs[provider][model]

  def unregister_all_models(self):
    self.model_registry = types.ModelRegistry(
        metadata=self.model_registry.metadata,
        default_model_priority_list=[],
        provider_model_configs={},
    )
    self.models_by_model_size = {}
    self.recommended_models = {}

  def override_default_model_priority_list(
      self,
      default_model_priority_list: list[types.ProviderModelType]):
    for provider_model in default_model_priority_list:
      if (provider_model.provider not in
          self.model_registry.provider_model_configs):
        raise ValueError(
            f'Provider {provider_model.provider} not registered.'
        )
      if (provider_model.model not in
          self.model_registry.provider_model_configs[provider_model.provider]):
        raise ValueError(
            f'Model {provider_model.model} not registered for provider '
            f'{provider_model.provider}.'
        )
    self.model_registry.default_model_priority_list = (
        default_model_priority_list
    )

#   def _validate_min_proxai_version(self, min_proxai_version: str | None):
#     if min_proxai_version is None:
#       return

#     current_version = version("proxai")

#     try:
#       specifier_set = SpecifierSet(min_proxai_version)
#       current = Version(current_version)

#       if not specifier_set.contains(current):
#         raise ValueError(
#             f'Current proxai version ({current_version}) does not satisfy '
#             f'the minimum version requirement: {min_proxai_version}. '
#             f'Please upgrade proxai to a version that satisfies this '
#             f'requirement.'
#         )
#     except InvalidSpecifier as e:
#       raise ValueError(
#           f'Model configs schema metadata min_proxai_version is invalid. '
#           f'Min proxai version specifier: {min_proxai_version}. '
#           f'Error: {e}'
#       ) from e
#     except InvalidVersion as e:
#       raise ValueError(
#           f'Current proxai version ({current_version}) is invalid. '
#           f'Error: {e}'
#       ) from e

#   def _validate_model_configs_schema_metadata(
#       self, model_configs_schema_metadata: types.ModelConfigsSchemaMetadataType
#   ):
#     self._validate_min_proxai_version(
#         model_configs_schema_metadata.min_proxai_version
#     )

#   def _get_provider_model_key(
#       self, provider_model: types.ProviderModelIdentifierType
#   ) -> tuple[str, str]:
#     """Extract (provider, model) tuple from any provider model identifier."""
#     if isinstance(provider_model, types.ProviderModelType):
#       return (provider_model.provider, provider_model.model)
#     elif self._is_provider_model_tuple(provider_model):
#       return (provider_model[0], provider_model[1])
#     raise ValueError(f'Invalid provider model identifier: {provider_model}')

#   def _get_all_recommended_models_from_configs(
#       self, provider_model_configs: types.ProviderModelConfigsType
#   ) -> set[tuple[str, str]]:
#     """Get (provider, model) tuples for all recommended models in configs."""
#     recommended = set()
#     for provider, models in provider_model_configs.items():
#       for model_name, config in models.items():
#         if config.metadata and config.metadata.is_recommended:
#           recommended.add((provider, model_name))
#     return recommended

#   def _get_all_models_by_call_type_from_configs(
#       self, provider_model_configs: types.ProviderModelConfigsType
#   ) -> dict[types.CallType, set[tuple[str, str]]]:
#     """Get models grouped by call_type from configs."""
#     by_call_type: dict[types.CallType, set[tuple[str, str]]] = {}
#     for provider, models in provider_model_configs.items():
#       for model_name, config in models.items():
#         if config.metadata and config.metadata.call_type:
#           call_type = config.metadata.call_type
#           if call_type not in by_call_type:
#             by_call_type[call_type] = set()
#           by_call_type[call_type].add((provider, model_name))
#     return by_call_type

#   def _get_all_models_by_size_from_configs(
#       self, provider_model_configs: types.ProviderModelConfigsType
#   ) -> dict[types.ModelSizeType, set[tuple[str, str]]]:
#     """Get models grouped by size from configs."""
#     by_size: dict[types.ModelSizeType, set[tuple[str, str]]] = {}
#     for provider, models in provider_model_configs.items():
#       for model_name, config in models.items():
#         if config.metadata and config.metadata.model_size_tags:
#           for size_tag in config.metadata.model_size_tags:
#             if size_tag not in by_size:
#               by_size[size_tag] = set()
#             by_size[size_tag].add((provider, model_name))
#     return by_size

#   def _validate_provider_model_key_matches_config(
#       self, provider_key: str, model_key: str,
#       config: types.ProviderModelConfigType
#   ):
#     """Validate provider_model fields match the dict keys."""
#     if config.provider_model is None:
#       raise ValueError(
#           f'provider_model is None for config at '
#           f'provider_model_configs[{provider_key}][{model_key}]'
#       )

#     if config.provider_model.provider != provider_key:
#       raise ValueError(
#           f'Provider mismatch: config key is "{provider_key}" but '
#           f'provider_model.provider is "{config.provider_model.provider}"'
#       )

#     if config.provider_model.model != model_key:
#       raise ValueError(
#           f'Model mismatch: config key is "{model_key}" but '
#           f'provider_model.model is "{config.provider_model.model}"'
#       )

#   def _validate_pricing(
#       self, provider_key: str, model_key: str,
#       pricing: types.ProviderModelPricingType
#   ):
#     """Validate pricing values are non-negative."""
#     if pricing is None:
#       raise ValueError(
#           f'pricing is None for '
#           f'provider_model_configs[{provider_key}][{model_key}]'
#       )

#     if (
#         pricing.per_query_token_cost is not None and
#         pricing.per_query_token_cost < 0
#     ):
#       raise ValueError(
#           f'per_query_token_cost is negative ({pricing.per_query_token_cost}) '
#           f'for provider_model_configs[{provider_key}][{model_key}]'
#       )

#     if (
#         pricing.per_response_token_cost is not None and
#         pricing.per_response_token_cost < 0
#     ):
#       raise ValueError(
#           f'per_response_token_cost is negative '
#           f'({pricing.per_response_token_cost}) for '
#           f'provider_model_configs[{provider_key}][{model_key}]'
#       )

#   def _validate_model_size_tags(
#       self, provider_key: str, model_key: str,
#       model_size_tags: list[types.ModelSizeType]
#   ):
#     """Validate model_size_tags contains only valid ModelSizeType values."""
#     valid_sizes = set(types.ModelSizeType)
#     for tag in model_size_tags:
#       if tag not in valid_sizes:
#         raise ValueError(
#             f'Invalid model_size_tag "{tag}" for '
#             f'provider_model_configs[{provider_key}][{model_key}]. '
#             f'Valid values: {[s.value for s in types.ModelSizeType]}'
#         )

#   def _validate_features(
#       self, provider_key: str, model_key: str,
#       features: types.ProviderModelFeatureType
#   ):
#     """Validate supported, best_effort, and not_supported are disjoint."""
#     if features is None:
#       return

#     for feature_name, feature in features.items():
#       supported = set(feature.supported or [])
#       best_effort = set(feature.best_effort or [])
#       not_supported = set(feature.not_supported or [])

#       supported_best_effort = supported & best_effort
#       if supported_best_effort:
#         raise ValueError(
#             f'Features {supported_best_effort} appear in both SUPPORTED and '
#             'BEST_EFFORT for provider_model_configs for '
#             f'({provider_key}, {model_key})\n'
#             f'Feature name: {feature_name}\n'
#             f'Feature config: {feature}'
#         )

#       supported_not_supported = supported & not_supported
#       if supported_not_supported:
#         raise ValueError(
#             f'Features {supported_not_supported} appear in both SUPPORTED and '
#             'NOT_SUPPORTED for provider_model_configs for '
#             f'({provider_key}, {model_key})\n'
#             f'Feature name: {feature_name}\n'
#             f'Feature config: {feature}'
#         )

#       best_effort_not_supported = best_effort & not_supported
#       if best_effort_not_supported:
#         raise ValueError(
#             f'Features {best_effort_not_supported} appear in both '
#             'BEST_EFFORT and NOT_SUPPORTED for provider_model_configs for '
#             f'({provider_key}, {model_key})\n'
#             f'Feature name: {feature_name}\n'
#             f'Feature config: {feature}'
#         )

#   def _validate_provider_model_config(
#       self, provider_key: str, model_key: str,
#       config: types.ProviderModelConfigType
#   ):
#     """Validate a single ProviderModelConfigType."""
#     self._validate_provider_model_key_matches_config(
#         provider_key, model_key, config
#     )

#     self._validate_pricing(provider_key, model_key, config.pricing)

#     self._validate_features(provider_key, model_key, config.features)

#     if (config.metadata and config.metadata.model_size_tags is not None):
#       self._validate_model_size_tags(
#           provider_key, model_key, config.metadata.model_size_tags
#       )

#   def _validate_provider_model_configs(
#       self, provider_model_configs: types.ProviderModelConfigsType
#   ):
#     """Validate all provider model configs."""
#     for provider_key, models in provider_model_configs.items():
#       for model_key, config in models.items():
#         self._validate_provider_model_config(provider_key, model_key, config)

#   def _validate_recommended_models(
#       self, provider_model_configs: types.ProviderModelConfigsType,
#       recommended_models: types.RecommendedModelsMappingType
#   ):
#     """Validate recommended_models matches is_recommended in configs."""
#     recommended_from_configs = self._get_all_recommended_models_from_configs(
#         provider_model_configs
#     )

#     recommended_from_list: set[tuple[str, str]] = set()
#     for _provider, models in recommended_models.items():
#       for model in models:
#         key = self._get_provider_model_key(model)
#         recommended_from_list.add(key)

#     missing_in_list = recommended_from_configs - recommended_from_list
#     if missing_in_list:
#       raise ValueError(
#           f'Models marked as is_recommended=True in provider_model_configs '
#           f'but missing from recommended_models: {sorted(missing_in_list)}'
#       )

#     extra_in_list = recommended_from_list - recommended_from_configs
#     if extra_in_list:
#       raise ValueError(
#           f'Models in recommended_models but not marked as is_recommended=True '
#           f'in provider_model_configs: {sorted(extra_in_list)}'
#       )

#   def _validate_models_by_call_type(
#       self, provider_model_configs: types.ProviderModelConfigsType,
#       models_by_call_type: types.ModelsByCallTypeType
#   ):
#     """Validate models_by_call_type matches call_type in configs."""
#     from_configs = self._get_all_models_by_call_type_from_configs(
#         provider_model_configs
#     )

#     from_list: dict[types.CallType, set[tuple[str, str]]] = {}
#     for call_type, providers in models_by_call_type.items():
#       from_list[call_type] = set()
#       for _provider, models in providers.items():
#         for model in models:
#           key = self._get_provider_model_key(model)
#           from_list[call_type].add(key)

#     all_call_types = set(from_configs.keys()) | set(from_list.keys())
#     for call_type in all_call_types:
#       config_models = from_configs.get(call_type, set())
#       list_models = from_list.get(call_type, set())

#       missing_in_list = config_models - list_models
#       if missing_in_list:
#         raise ValueError(
#             f'Models with call_type={call_type} in provider_model_configs '
#             f'but missing from models_by_call_type: {sorted(missing_in_list)}'
#         )

#       extra_in_list = list_models - config_models
#       if extra_in_list:
#         raise ValueError(
#             f'Models in models_by_call_type[{call_type}] but not marked with '
#             f'that call_type in configs: {sorted(extra_in_list)}'
#         )

#   def _validate_models_by_size(
#       self, provider_model_configs: types.ProviderModelConfigsType,
#       models_by_size: types.ModelsBySizeType
#   ):
#     """Validate models_by_size matches model_size_tags in configs."""
#     from_configs = self._get_all_models_by_size_from_configs(
#         provider_model_configs
#     )

#     from_list: dict[types.ModelSizeType, set[tuple[str, str]]] = {}
#     for size, models in models_by_size.items():
#       from_list[size] = set()
#       for model in models:
#         key = self._get_provider_model_key(model)
#         from_list[size].add(key)

#     all_sizes = set(from_configs.keys()) | set(from_list.keys())
#     for size in all_sizes:
#       config_models = from_configs.get(size, set())
#       list_models = from_list.get(size, set())

#       missing_in_list = config_models - list_models
#       if missing_in_list:
#         raise ValueError(
#             f'Models with model_size_tags containing {size} in '
#             f'provider_model_configs but missing from models_by_size: '
#             f'{sorted(missing_in_list)}'
#         )

#       extra_in_list = list_models - config_models
#       if extra_in_list:
#         raise ValueError(
#             f'Models in models_by_size[{size}] but model_size_tags in '
#             f'provider_model_configs does not contain {size}: '
#             f'{sorted(extra_in_list)}'
#         )

#   def _validate_default_model_priority_list(
#       self, provider_model_configs: types.ProviderModelConfigsType,
#       default_model_priority_list: types.DefaultModelPriorityListType
#   ):
#     """Validate all models in default_model_priority_list exist in configs."""
#     for model in default_model_priority_list:
#       key = self._get_provider_model_key(model)
#       provider, model_name = key
#       if provider not in provider_model_configs:
#         raise ValueError(
#             f'Provider {provider} in default_model_priority_list '
#             f'not found in provider_model_configs'
#         )
#       if model_name not in provider_model_configs[provider]:
#         raise ValueError(
#             f'Model {model_name} for provider {provider} in '
#             f'default_model_priority_list not found in provider_model_configs'
#         )

#   def _validate_version_config(
#       self, version_config: types.ModelConfigsSchemaVersionConfigType
#   ):
#     """Validate version_config internal consistency."""
#     provider_model_configs = version_config.provider_model_configs
#     if provider_model_configs is None:
#       return

#     self._validate_provider_model_configs(provider_model_configs)

#     if version_config.recommended_models is not None:
#       self._validate_recommended_models(
#           provider_model_configs, version_config.recommended_models
#       )

#     if version_config.models_by_call_type is not None:
#       self._validate_models_by_call_type(
#           provider_model_configs, version_config.models_by_call_type
#       )

#     if version_config.models_by_size is not None:
#       self._validate_models_by_size(
#           provider_model_configs, version_config.models_by_size
#       )

#     if version_config.default_model_priority_list is not None:
#       self._validate_default_model_priority_list(
#           provider_model_configs, version_config.default_model_priority_list
#       )

#   def _validate_model_configs_schema(
#       self, model_configs_schema: types.ModelConfigsSchemaType
#   ):
#     if model_configs_schema.metadata:
#       self._validate_model_configs_schema_metadata(
#           model_configs_schema.metadata
#       )

#     if model_configs_schema.version_config:
#       self._validate_version_config(model_configs_schema.version_config)

  def _is_provider_model_tuple(self, value: Any) -> bool:
    return (
        isinstance(value, tuple) and len(value) == 2 and
        isinstance(value[0], str) and isinstance(value[1], str)
    )

  @staticmethod
  def _load_model_registry_from_local_files(
      version: str | None = None
  ) -> types.ModelRegistry:
    """Load model registry from bundled JSON files."""
    version = version or ModelConfigs.LOCAL_CONFIG_VERSION

    try:
      config_data = (
          resources.files("proxai.connectors.model_configs_data").
          joinpath(f"{version}.json").read_text(encoding="utf-8")
      )
    except FileNotFoundError as e:
      raise FileNotFoundError(
          f'Model config file "{version}.json" not found in package. '
          'Please update the proxai package to the latest version. '
          'If updating does not resolve the issue, please contact '
          'support@proxai.co'
      ) from e

    try:
      config_dict = json.loads(config_data)
    except json.JSONDecodeError as e:
      raise ValueError(
          f'Invalid JSON in config file "{version}.json". '
          'Please update the proxai package to the latest version. '
          'If updating does not resolve the issue, please contact '
          f'support@proxai.co\nError: {e}'
      ) from e

    return type_serializer.decode_model_registry(config_dict)

  def load_model_registry_from_json_string(self, json_string: str):
    """Load model registry from a JSON string."""
    model_registry = type_serializer.decode_model_registry(
        json.loads(json_string)
    )
    self.model_registry = model_registry

  def check_provider_model_identifier_type(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType,
  ):
    """Check if provider model identifier is supported."""
    if isinstance(provider_model_identifier, types.ProviderModelType):
      provider = provider_model_identifier.provider
      model = provider_model_identifier.model
      if provider not in self.model_registry.provider_model_configs:
        raise ValueError(
            f'Provider not supported: {provider}.\n'
            'Supported providers: '
            f'{list(self.model_registry.provider_model_configs.keys())}'
        )
      if model not in self.model_registry.provider_model_configs[provider]:
        raise ValueError(
            f'Model not supported: {model}.\nSupported models: '
            f'{self.model_registry.provider_model_configs[provider].keys()}'
        )
      config_provider_model = (
          self.model_registry.provider_model_configs[
              provider][model].provider_model)
      if provider_model_identifier != config_provider_model:
        raise ValueError(
            'Mismatch between provider model identifier and model config.'
            f'Provider model identifier: {provider_model_identifier}'
            f'Model config: {config_provider_model}'
        )
    elif self._is_provider_model_tuple(provider_model_identifier):
      provider = provider_model_identifier[0]
      model = provider_model_identifier[1]
      if provider not in self.model_registry.provider_model_configs:
        raise ValueError(
            f'Provider not supported: {provider}.\n'
            'Supported providers: '
            f'{self.model_registry.provider_model_configs.keys()}'
        )
      if model not in self.model_registry.provider_model_configs[provider]:
        raise ValueError(
            f'Model not supported: {model}.\nSupported models: '
            f'{self.model_registry.provider_model_configs[provider].keys()}'
        )
    else:
      raise ValueError(
          f'Invalid provider model identifier: {provider_model_identifier}'
      )

  def get_provider_model(
      self, model_identifier: types.ProviderModelIdentifierType
  ) -> types.ProviderModelType:
    """Convert a model identifier to a ProviderModelType."""
    if self._is_provider_model_tuple(model_identifier):
      return self.model_registry.provider_model_configs[
          model_identifier[0]][model_identifier[1]].provider_model
    else:
      provider_model = self.model_registry.provider_model_configs[
          model_identifier.provider][model_identifier.model].provider_model
      if (provider_model.provider_model_identifier !=
          model_identifier.provider_model_identifier):
        raise ValueError(
            'Provider model identifier mismatch: '
            f'{provider_model.provider_model_identifier} != '
            f'{model_identifier.provider_model_identifier}')
      return model_identifier

  def get_provider_model_config(
      self, model_identifier: types.ProviderModelIdentifierType
  ) -> types.ProviderModelType:
    """Get the full config for a model identifier."""
    provider_model = self.get_provider_model(model_identifier)
    return self.model_registry.provider_model_configs[
        provider_model.provider][provider_model.model]

#   def get_provider_model_cost(
#       self,
#       provider_model_identifier: types.ProviderModelIdentifierType,
#       query_token_count: int,
#       response_token_count: int,
#   ) -> int:
#     """Calculate the cost in micro-cents for a query."""
#     provider_model = self.get_provider_model(provider_model_identifier)
#     version_config = self.model_configs_schema.version_config
#     model_pricing_config = version_config.provider_model_configs[
#         provider_model.provider][provider_model.model].pricing
#     return math.floor(
#         query_token_count * model_pricing_config.per_query_token_cost +
#         response_token_count * model_pricing_config.per_response_token_cost
#     )

  def get_all_models(
      self,
      provider: types.ProviderNameType | None = None,
      model_size: types.ModelSizeType | None = None,
      recommended_only: bool | None = True,
  ) -> list[types.ProviderModelType]:
    """List all models matching the given filters."""
    if (
        model_size is not None and
        model_size not in self.models_by_model_size
    ):
      raise ValueError(f'Model size not supported: {model_size}')

    if (
        provider is not None and
        provider not in self.model_registry.provider_model_configs
    ):
      supported_providers = list(
          self.model_registry.provider_model_configs.keys())
      raise ValueError(
          f'Provider not supported: {provider}.\n'
          f'Supported providers: {supported_providers}'
      )

    result_provider_models = []
    for provider_name, provider_models in (
        self.model_registry.provider_model_configs.items()
    ):
      if provider is not None and provider_name != provider:
        continue

      for provider_model_config in provider_models.values():
        if (
            model_size is not None and (
                provider_model_config.metadata.model_size_tags is None or
                model_size not in
                provider_model_config.metadata.model_size_tags
            )
        ):
          continue

        if (recommended_only and
            not provider_model_config.metadata.is_recommended):
          continue

        result_provider_models.append(
            provider_model_config.provider_model)

    return result_provider_models

  def export_to_json(self, file_path: str):
    """Export model registry to a JSON file with sorted keys."""
    record = type_serializer.encode_model_registry(self.model_registry)

    # Sort provider_model_configs recursively.
    if 'provider_model_configs' in record:
      record['provider_model_configs'] = self._sort_value(
          record['provider_model_configs']
      )

    # Enforce key ordering.
    ordered_record = {}
    key_order = [
        'metadata', 'default_model_priority_list', 'provider_model_configs'
    ]
    for key in key_order:
      if key in record:
        ordered_record[key] = record[key]
    for key in record:
      if key not in ordered_record:
        ordered_record[key] = record[key]

    with open(file_path, 'w') as f:
      json.dump(ordered_record, f, indent=2)
      f.write('\n')

  @staticmethod
  def _sort_value(value):
    """Recursively sort dict keys and list values."""
    if isinstance(value, dict):
      return {
          k: ModelConfigs._sort_value(v)
          for k, v in sorted(value.items())
      }
    elif isinstance(value, list):
      sorted_items = [ModelConfigs._sort_value(item) for item in value]
      try:
        return sorted(
            sorted_items,
            key=lambda x: json.dumps(x, sort_keys=True)
        )
      except TypeError:
        return sorted_items
    return value

  def get_default_model_priority_list(self) -> list[types.ProviderModelType]:
    """Return the default model priority list for fallback selection."""
    return self.model_registry.default_model_priority_list

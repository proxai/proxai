from __future__ import annotations

import dataclasses
import json
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
  _model_configs_state: types.ModelConfigsState | None

  LOCAL_CONFIG_VERSION = "v1.3.0"

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

    del self.model_registry.provider_model_configs[provider][model]

  def unregister_all_models(self):
    self.model_registry = types.ModelRegistry(
        metadata=self.model_registry.metadata,
        default_model_priority_list=[],
        provider_model_configs={},
    )

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

  @staticmethod
  def _validate_min_proxai_version(min_proxai_version: str | None):
    if min_proxai_version is None:
      return
    current_version = version("proxai")
    try:
      specifier_set = SpecifierSet(min_proxai_version)
      current = Version(current_version)
      if not specifier_set.contains(current):
        raise ValueError(
            f'Current proxai version ({current_version}) does not satisfy '
            f'the minimum version requirement: {min_proxai_version}. '
            f'Please upgrade proxai to a version that satisfies this '
            f'requirement.'
        )
    except InvalidSpecifier as e:
      raise ValueError(
          f'Model configs schema metadata min_proxai_version is invalid. '
          f'Min proxai version specifier: {min_proxai_version}. '
          f'Error: {e}'
      ) from e
    except InvalidVersion as e:
      raise ValueError(
          f'Current proxai version ({current_version}) is invalid. '
          f'Error: {e}'
      ) from e

  @staticmethod
  def _validate_provider_model_configs(model_registry: types.ModelRegistry):
    """Validate per-model invariants: key-match and non-negative pricing.

    Raised errors are re-raised by reload_from_registry wrapped in a
    'Failed to load model registry' message so the stack trace is readable
    for users who are loading a registry from JSON / ProxDash.
    """
    provider_model_configs = model_registry.provider_model_configs or {}
    for provider, models in provider_model_configs.items():
      for model, config in models.items():
        if config.provider_model is None:
          raise ValueError(
              f'provider_model is None for '
              f'provider_model_configs[{provider!r}][{model!r}]'
          )
        if config.provider_model.provider != provider:
          raise ValueError(
              f'Provider key mismatch at '
              f'provider_model_configs[{provider!r}][{model!r}]: '
              f'provider_model.provider is '
              f'{config.provider_model.provider!r}'
          )
        if config.provider_model.model != model:
          raise ValueError(
              f'Model key mismatch at '
              f'provider_model_configs[{provider!r}][{model!r}]: '
              f'provider_model.model is {config.provider_model.model!r}'
          )
        if config.pricing is not None:
          if (config.pricing.input_token_cost is not None and
              config.pricing.input_token_cost < 0):
            raise ValueError(
                f'input_token_cost is negative '
                f'({config.pricing.input_token_cost}) at '
                f'provider_model_configs[{provider!r}][{model!r}]'
            )
          if (config.pricing.output_token_cost is not None and
              config.pricing.output_token_cost < 0):
            raise ValueError(
                f'output_token_cost is negative '
                f'({config.pricing.output_token_cost}) at '
                f'provider_model_configs[{provider!r}][{model!r}]'
            )

  @staticmethod
  def _validate_default_model_priority_list(
      model_registry: types.ModelRegistry):
    if not model_registry.default_model_priority_list:
      return
    provider_model_configs = model_registry.provider_model_configs or {}
    for entry in model_registry.default_model_priority_list:
      if entry.provider not in provider_model_configs:
        raise ValueError(
            f'Provider {entry.provider!r} in default_model_priority_list '
            f'not found in provider_model_configs'
        )
      if entry.model not in provider_model_configs[entry.provider]:
        raise ValueError(
            f'Model {entry.model!r} for provider {entry.provider!r} in '
            f'default_model_priority_list not found in provider_model_configs'
        )

  def reload_from_registry(self, model_registry: types.ModelRegistry):
    """Replace the current registry, validating business invariants.

    Skips the min-proxai-version gate for BUILT_IN configs (the bundled JSON
    is trusted by construction); enforces it for PROXDASH and other sources.
    Any validation failure is re-raised with a 'Failed to load model registry'
    prefix so callers see one clear error at the load boundary, not deep
    inside deserialization.
    """
    try:
      if model_registry.metadata is not None:
        config_origin = model_registry.metadata.config_origin
        if config_origin != types.ConfigOriginType.BUILT_IN:
          self._validate_min_proxai_version(
              model_registry.metadata.min_proxai_version
          )
      self._validate_provider_model_configs(model_registry)
      self._validate_default_model_priority_list(model_registry)
    except ValueError as e:
      raise ValueError(f'Failed to load model registry: {e}') from e
    self.model_registry = model_registry

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
    self.reload_from_registry(model_registry)

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

  def get_all_models(
      self,
      provider: types.ProviderNameType | None = None,
      model_size: types.ModelSizeType | None = None,
      recommended_only: bool | None = True,
  ) -> list[types.ProviderModelType]:
    """List all models matching the given filters."""
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

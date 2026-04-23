"""Tests for ModelConfigs."""

from importlib import resources

import pytest

import proxai.connectors.model_configs as model_configs
import proxai.types as types


# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture
def shared_configs():
  """Read-only shared instance from conftest.

  Loaded from example_proxdash_model_configs.json (openai + gemini providers)
  plus the three mock_* providers registered programmatically in conftest.
  """
  return pytest.model_configs_instance


@pytest.fixture
def fresh_configs():
  """Fresh ModelConfigs for tests that mutate the registry."""
  return model_configs.ModelConfigs()


def _make_config(provider, model, identifier=None):
  """Minimal valid ProviderModelConfig."""
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider, model=model,
          provider_model_identifier=identifier or model),
      pricing=types.ProviderModelPricingType(
          input_token_cost_nano_usd_per_token=1, output_token_cost_nano_usd_per_token=1),
      features=types.FeatureConfigType(),
      metadata=types.ProviderModelMetadataType(),
  )


# =============================================================================
# Helpers
# =============================================================================


class TestIsProviderModelTuple:

  def test_valid_tuple(self, shared_configs):
    assert shared_configs._is_provider_model_tuple(
        ('openai', 'gpt-4o')) is True

  def test_invalid_inputs(self, shared_configs):
    # Wrong length.
    assert shared_configs._is_provider_model_tuple(('openai',)) is False
    # Non-string items.
    assert shared_configs._is_provider_model_tuple((1, 'gpt-4')) is False
    # Not a tuple.
    assert shared_configs._is_provider_model_tuple(
        ['openai', 'gpt-4']) is False


# =============================================================================
# check_provider_model_identifier_type
# =============================================================================


class TestCheckProviderModelIdentifierType:

  def test_valid_passes(self, shared_configs):
    pm = shared_configs.get_provider_model(('openai', 'gpt-4o'))
    shared_configs.check_provider_model_identifier_type(pm)
    shared_configs.check_provider_model_identifier_type(('openai', 'gpt-4o'))

  def test_unknown_provider_raises(self, shared_configs):
    with pytest.raises(ValueError, match='Provider not supported'):
      shared_configs.check_provider_model_identifier_type(('unknown', 'x'))

  def test_unknown_model_raises(self, shared_configs):
    with pytest.raises(ValueError, match='Model not supported'):
      shared_configs.check_provider_model_identifier_type(
          ('openai', 'unknown'))


# =============================================================================
# Query methods
# =============================================================================


class TestGetProviderModel:

  def test_with_tuple(self, shared_configs):
    pm = shared_configs.get_provider_model(('openai', 'gpt-4o'))
    assert isinstance(pm, types.ProviderModelType)
    assert pm.provider == 'openai'
    assert pm.model == 'gpt-4o'

  def test_with_provider_model_type_returns_same(self, shared_configs):
    pm = shared_configs.get_provider_model(('openai', 'gpt-4o'))
    assert shared_configs.get_provider_model(pm) == pm


class TestGetProviderModelConfig:

  def test_returns_config(self, shared_configs):
    config = shared_configs.get_provider_model_config(('openai', 'gpt-4o'))
    assert isinstance(config, types.ProviderModelConfig)
    assert config.provider_model.provider == 'openai'
    assert config.provider_model.model == 'gpt-4o'


class TestGetAllModels:

  def test_recommended_only_default(self, shared_configs):
    result = shared_configs.get_all_models()
    assert len(result) > 0
    for pm in result:
      config = shared_configs.get_provider_model_config(pm)
      assert config.metadata.is_recommended is True

  def test_model_size_filter(self, shared_configs):
    # Mock providers have SMALL tag, so the result is non-empty.
    result = shared_configs.get_all_models(
        model_size=types.ModelSizeType.SMALL, recommended_only=False)
    assert len(result) > 0
    for pm in result:
      config = shared_configs.get_provider_model_config(pm)
      assert types.ModelSizeType.SMALL in config.metadata.model_size_tags

  def test_invalid_provider_raises(self, shared_configs):
    with pytest.raises(ValueError, match='Provider not supported'):
      shared_configs.get_all_models(provider='nonexistent')


class TestGetDefaultModelPriorityList:

  def test_returns_list(self, shared_configs):
    result = shared_configs.get_default_model_priority_list()
    assert isinstance(result, list)
    for pm in result:
      assert isinstance(pm, types.ProviderModelType)


# =============================================================================
# Registry mutation
# =============================================================================


class TestRegisterProviderModelConfig:

  def test_registers_new_model(self, fresh_configs):
    config = _make_config('testp', 'testm')
    fresh_configs.register_provider_model_config(config)
    assert fresh_configs.get_provider_model(
        ('testp', 'testm')).model == 'testm'

  def test_duplicate_raises(self, fresh_configs):
    config = _make_config('testp', 'testm')
    fresh_configs.register_provider_model_config(config)
    with pytest.raises(ValueError, match='already registered'):
      fresh_configs.register_provider_model_config(config)


class TestUnregisterModel:

  def test_removes_registered_model(self, fresh_configs):
    config = _make_config('testp', 'testm')
    fresh_configs.register_provider_model_config(config)
    fresh_configs.unregister_model(config.provider_model)
    provider_models = (
        fresh_configs.model_registry.provider_model_configs.get('testp', {}))
    assert 'testm' not in provider_models

  def test_unknown_provider_raises(self, fresh_configs):
    pm = types.ProviderModelType(
        provider='nonexistent', model='x', provider_model_identifier='x')
    with pytest.raises(ValueError, match='Provider .* not registered'):
      fresh_configs.unregister_model(pm)

  def test_identifier_mismatch_raises(self, fresh_configs):
    config = _make_config('testp', 'testm', identifier='id-1')
    fresh_configs.register_provider_model_config(config)
    wrong = types.ProviderModelType(
        provider='testp', model='testm', provider_model_identifier='id-2')
    with pytest.raises(
        ValueError, match='Provider model identifier mismatch'):
      fresh_configs.unregister_model(wrong)


class TestUnregisterAllModels:

  def test_clears_all_and_priority_list(self, fresh_configs):
    fresh_configs.unregister_all_models()
    assert fresh_configs.model_registry.provider_model_configs == {}
    assert fresh_configs.model_registry.default_model_priority_list == []


class TestOverrideDefaultModelPriorityList:

  def test_replaces_priority_list(self, fresh_configs):
    config = _make_config('testp', 'testm')
    fresh_configs.register_provider_model_config(config)
    fresh_configs.override_default_model_priority_list([config.provider_model])
    assert fresh_configs.get_default_model_priority_list() == [
        config.provider_model]

  def test_unknown_model_raises(self, fresh_configs):
    unknown = types.ProviderModelType(
        provider='openai', model='nonexistent-model',
        provider_model_identifier='x')
    with pytest.raises(ValueError, match='not registered'):
      fresh_configs.override_default_model_priority_list([unknown])


# =============================================================================
# Validation
# =============================================================================


class TestValidateProviderModelConfigs:

  def test_valid_passes(self, shared_configs):
    model_configs.ModelConfigs._validate_provider_model_configs(
        shared_configs.model_registry)

  def test_provider_key_mismatch_raises(self):
    registry = types.ModelRegistry(
        metadata=None, default_model_priority_list=[],
        provider_model_configs={
            'openai': {'gpt-4': _make_config('wrong_provider', 'gpt-4')},
        })
    with pytest.raises(ValueError, match='Provider key mismatch'):
      model_configs.ModelConfigs._validate_provider_model_configs(registry)

  def test_negative_pricing_raises(self):
    config = types.ProviderModelConfig(
        provider_model=types.ProviderModelType(
            provider='openai', model='gpt-4',
            provider_model_identifier='gpt-4'),
        pricing=types.ProviderModelPricingType(input_token_cost_nano_usd_per_token=-1),
        features=types.FeatureConfigType(),
        metadata=types.ProviderModelMetadataType())
    registry = types.ModelRegistry(
        metadata=None, default_model_priority_list=[],
        provider_model_configs={'openai': {'gpt-4': config}})
    with pytest.raises(ValueError, match='input_token_cost_nano_usd_per_token is negative'):
      model_configs.ModelConfigs._validate_provider_model_configs(registry)


class TestValidateDefaultModelPriorityList:

  def test_valid_passes(self, shared_configs):
    model_configs.ModelConfigs._validate_default_model_priority_list(
        shared_configs.model_registry)

  def test_unknown_model_raises(self):
    registry = types.ModelRegistry(
        metadata=None,
        default_model_priority_list=[
            types.ProviderModelType(
                provider='openai', model='nonexistent',
                provider_model_identifier='x'),
        ],
        provider_model_configs={'openai': {}})
    with pytest.raises(
        ValueError, match='default_model_priority_list not found'):
      model_configs.ModelConfigs._validate_default_model_priority_list(
          registry)


class TestValidateMinProxaiVersion:

  def test_none_skips(self):
    model_configs.ModelConfigs._validate_min_proxai_version(None)

  def test_satisfied_passes(self):
    model_configs.ModelConfigs._validate_min_proxai_version('>=0.0.1')

  def test_not_satisfied_raises(self):
    with pytest.raises(ValueError, match='does not satisfy'):
      model_configs.ModelConfigs._validate_min_proxai_version('>=999.0.0')


# =============================================================================
# reload_from_registry / load_model_registry_from_json_string
# =============================================================================


class TestReloadFromRegistry:

  def test_happy_path_replaces_registry(self, fresh_configs):
    new_registry = types.ModelRegistry(
        metadata=None, default_model_priority_list=[],
        provider_model_configs={
            'myp': {'mym': _make_config('myp', 'mym')},
        })
    fresh_configs.reload_from_registry(new_registry)
    assert set(fresh_configs.model_registry.provider_model_configs.keys()) == {
        'myp'}

  def test_validation_error_wrapped(self, fresh_configs):
    bad = types.ModelRegistry(
        metadata=None, default_model_priority_list=[],
        provider_model_configs={
            'openai': {'gpt-4': _make_config('wrong', 'gpt-4')},
        })
    with pytest.raises(ValueError, match='Failed to load model registry'):
      fresh_configs.reload_from_registry(bad)


class TestLoadModelRegistryFromJsonString:

  def test_loads_example_proxdash_config(self, fresh_configs):
    data = resources.files(
        "proxai.connectors.model_configs_data"
    ).joinpath("example_proxdash_model_configs.json").read_text(
        encoding="utf-8")
    fresh_configs.load_model_registry_from_json_string(data)
    configs = fresh_configs.model_registry.provider_model_configs
    assert 'openai' in configs
    assert 'gemini' in configs


# =============================================================================
# export_to_json
# =============================================================================


class TestExportToJson:

  def test_round_trip(self, fresh_configs, tmp_path):
    file_path = tmp_path / 'export.json'
    fresh_configs.export_to_json(str(file_path))

    restored = model_configs.ModelConfigs()
    restored.load_model_registry_from_json_string(
        file_path.read_text(encoding='utf-8'))
    assert restored.model_registry == fresh_configs.model_registry


# =============================================================================
# Built-in config
# =============================================================================


class TestBuiltInConfigLoads:

  def test_bundled_config_passes_validators(self, fresh_configs):
    model_configs.ModelConfigs._validate_provider_model_configs(
        fresh_configs.model_registry)
    model_configs.ModelConfigs._validate_default_model_priority_list(
        fresh_configs.model_registry)

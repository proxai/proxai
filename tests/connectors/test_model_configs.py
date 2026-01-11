import pytest
import proxai.types as types
import proxai.connectors.model_configs as model_configs


# =============================================================================
# Test Helpers
# =============================================================================

def create_provider_model_config(
    provider: str,
    model: str,
    is_featured: bool = True,
    call_type: types.CallType = types.CallType.GENERATE_TEXT,
    model_size_tags: list = None,
    features: dict = None
) -> types.ProviderModelConfigType:
  """Create a ProviderModelConfigType for testing."""
  return types.ProviderModelConfigType(
      provider_model=types.ProviderModelType(
          provider=provider,
          model=model,
          provider_model_identifier=f'{model}-identifier'),
      pricing=types.ProviderModelPricingType(
          per_response_token_cost=1.0,
          per_query_token_cost=1.0),
      features=features or {},
      metadata=types.ProviderModelMetadataType(
          call_type=call_type,
          is_featured=is_featured,
          model_size_tags=model_size_tags))


def create_feature_config(
    supported: list = None,
    best_effort: list = None,
    not_supported: list = None
) -> types.EndpointFeatureInfoType:
  """Create an EndpointFeatureInfoType for testing."""
  return types.EndpointFeatureInfoType(
      supported=supported or [],
      best_effort=best_effort or [],
      not_supported=not_supported or [])


def create_version_config(
    provider_model_configs: dict,
    featured_models: dict = None,
    models_by_call_type: dict = None,
    models_by_size: dict = None,
    default_model_priority_list: list = None
) -> types.ModelConfigsSchemaVersionConfigType:
  """Create a ModelConfigsSchemaVersionConfigType for testing."""
  return types.ModelConfigsSchemaVersionConfigType(
      provider_model_configs=provider_model_configs,
      featured_models=featured_models or {},
      models_by_call_type=models_by_call_type or {},
      models_by_size=models_by_size or {},
      default_model_priority_list=default_model_priority_list or [])


def create_schema(
    version_config: types.ModelConfigsSchemaVersionConfigType
) -> types.ModelConfigsSchemaType:
  """Create a ModelConfigsSchemaType for testing."""
  return types.ModelConfigsSchemaType(
      metadata=types.ModelConfigsSchemaMetadataType(version='1.0.0'),
      version_config=version_config)


@pytest.fixture
def model_configs_instance():
  """Fixture to provide a ModelConfigs instance."""
  return model_configs.ModelConfigs()


# =============================================================================
# Basic Utility Tests
# =============================================================================

class TestIsProviderModelTuple:
  def test_valid_tuple(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(('openai', 'gpt-4')) is True

  def test_invalid_length(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(('openai',)) is False

  def test_invalid_types(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple((1, 'gpt-4')) is False

  def test_not_tuple(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(['openai', 'gpt-4']) is False


class TestGetProviderModelKey:
  def test_with_provider_model_type(self, model_configs_instance):
    pm = types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')
    assert model_configs_instance._get_provider_model_key(pm) == ('openai', 'gpt-4')

  def test_with_tuple(self, model_configs_instance):
    assert model_configs_instance._get_provider_model_key(('claude', 'opus')) == ('claude', 'opus')

  def test_invalid_type_raises(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance._get_provider_model_key('invalid')


# =============================================================================
# Provider Model Identifier Tests
# =============================================================================

class TestCheckProviderModelIdentifierType:
  def test_unsupported_provider_raises(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          types.ProviderModelType(provider='invalid', model='model', provider_model_identifier='id'))

  def test_unsupported_model_raises(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          types.ProviderModelType(provider='openai', model='invalid', provider_model_identifier='id'))

  def test_valid_model_passes(self, model_configs_instance):
    provider_model = model_configs_instance.get_provider_model(('claude', 'opus-4'))
    model_configs_instance.check_provider_model_identifier_type(provider_model)

  def test_valid_tuple_passes(self, model_configs_instance):
    model_configs_instance.check_provider_model_identifier_type(('claude', 'opus-4'))


# =============================================================================
# Config Extraction Tests
# =============================================================================

class TestGetAllFeaturedModelsFromConfigs:
  def test_returns_featured_models(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_provider_model_config('openai', 'gpt-4', is_featured=True),
            'gpt-3': create_provider_model_config('openai', 'gpt-3', is_featured=False),
        }
    }
    result = model_configs_instance._get_all_featured_models_from_configs(configs)
    assert result == {('openai', 'gpt-4')}

  def test_empty_configs(self, model_configs_instance):
    result = model_configs_instance._get_all_featured_models_from_configs({})
    assert result == set()


class TestGetAllModelsByCallTypeFromConfigs:
  def test_groups_by_call_type(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_provider_model_config('openai', 'gpt-4', call_type=types.CallType.GENERATE_TEXT),
        }
    }
    result = model_configs_instance._get_all_models_by_call_type_from_configs(configs)
    assert result[types.CallType.GENERATE_TEXT] == {('openai', 'gpt-4')}


class TestGetAllModelsBySizeFromConfigs:
  def test_groups_by_size(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_provider_model_config('openai', 'gpt-4', model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }
    result = model_configs_instance._get_all_models_by_size_from_configs(configs)
    assert result[types.ModelSizeType.LARGE] == {('openai', 'gpt-4')}

  def test_model_with_multiple_size_tags(self, model_configs_instance):
    """Model with multiple size tags appears in all corresponding groups."""
    configs = {
        'openai': {
            'gpt-4': create_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[types.ModelSizeType.LARGE, types.ModelSizeType.LARGEST]),
        }
    }
    result = model_configs_instance._get_all_models_by_size_from_configs(configs)
    assert ('openai', 'gpt-4') in result[types.ModelSizeType.LARGE]
    assert ('openai', 'gpt-4') in result[types.ModelSizeType.LARGEST]


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateProviderModelKeyMatchesConfig:
  def test_valid_config(self, model_configs_instance):
    config = create_provider_model_config('openai', 'gpt-4')
    model_configs_instance._validate_provider_model_key_matches_config('openai', 'gpt-4', config)

  def test_provider_mismatch_raises(self, model_configs_instance):
    config = create_provider_model_config('claude', 'gpt-4')
    with pytest.raises(ValueError, match='Provider mismatch'):
      model_configs_instance._validate_provider_model_key_matches_config('openai', 'gpt-4', config)

  def test_model_mismatch_raises(self, model_configs_instance):
    config = create_provider_model_config('openai', 'gpt-3')
    with pytest.raises(ValueError, match='Model mismatch'):
      model_configs_instance._validate_provider_model_key_matches_config('openai', 'gpt-4', config)


class TestValidatePricing:
  def test_valid_pricing(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(per_query_token_cost=1.0, per_response_token_cost=2.0)
    model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)

  def test_none_pricing_raises(self, model_configs_instance):
    with pytest.raises(ValueError, match='pricing is None'):
      model_configs_instance._validate_pricing('openai', 'gpt-4', None)

  def test_negative_cost_raises(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(per_query_token_cost=-1.0, per_response_token_cost=2.0)
    with pytest.raises(ValueError, match='negative'):
      model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)


class TestValidateModelSizeTags:
  def test_valid_size_tags(self, model_configs_instance):
    model_configs_instance._validate_model_size_tags('openai', 'gpt-4', [types.ModelSizeType.LARGE])

  def test_invalid_size_tag_raises(self, model_configs_instance):
    with pytest.raises(ValueError, match='Invalid model_size_tag'):
      model_configs_instance._validate_model_size_tags('openai', 'gpt-4', ['invalid_size'])


class TestValidateFeatures:
  """Tests for _validate_features - validates endpoint lists are disjoint per feature."""

  def test_none_features_valid(self, model_configs_instance):
    model_configs_instance._validate_features('openai', 'gpt-4', None)

  def test_empty_features_valid(self, model_configs_instance):
    model_configs_instance._validate_features('openai', 'gpt-4', {})

  def test_disjoint_endpoints_valid(self, model_configs_instance):
    features = {
        'prompt': create_feature_config(
            supported=['chat'],
            best_effort=['completion'],
            not_supported=['legacy'])
    }
    model_configs_instance._validate_features('openai', 'gpt-4', features)

  def test_supported_best_effort_overlap_raises(self, model_configs_instance):
    features = {
        'prompt': create_feature_config(
            supported=['chat', 'completion'],
            best_effort=['completion'])  # 'completion' overlaps
    }
    with pytest.raises(ValueError, match='SUPPORTED and BEST_EFFORT'):
      model_configs_instance._validate_features('openai', 'gpt-4', features)

  def test_supported_not_supported_overlap_raises(self, model_configs_instance):
    features = {
        'prompt': create_feature_config(
            supported=['chat'],
            not_supported=['chat'])  # 'chat' overlaps
    }
    with pytest.raises(ValueError, match='SUPPORTED and NOT_SUPPORTED'):
      model_configs_instance._validate_features('openai', 'gpt-4', features)

  def test_best_effort_not_supported_overlap_raises(self, model_configs_instance):
    features = {
        'prompt': create_feature_config(
            best_effort=['chat'],
            not_supported=['chat'])  # 'chat' overlaps
    }
    with pytest.raises(ValueError, match='BEST_EFFORT and NOT_SUPPORTED'):
      model_configs_instance._validate_features('openai', 'gpt-4', features)


class TestValidateFeaturedModels:
  def test_valid_featured_models(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', is_featured=True)}
    }
    featured = {
        'openai': [types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')]
    }
    model_configs_instance._validate_featured_models(configs, featured)

  def test_missing_in_list_raises(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', is_featured=True)}
    }
    with pytest.raises(ValueError, match='missing from featured_models'):
      model_configs_instance._validate_featured_models(configs, {})

  def test_extra_in_list_raises(self, model_configs_instance):
    """Detects when featured_models has entries not marked is_featured=True in config."""
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', is_featured=False)}
    }
    featured = {
        'openai': [types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')]
    }
    with pytest.raises(ValueError, match='not marked as is_featured=True'):
      model_configs_instance._validate_featured_models(configs, featured)


class TestValidateModelsByCallType:
  def test_valid_models_by_call_type(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', call_type=types.CallType.GENERATE_TEXT)}
    }
    by_call_type = {
        types.CallType.GENERATE_TEXT: {
            'openai': [types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')]
        }
    }
    model_configs_instance._validate_models_by_call_type(configs, by_call_type)

  def test_missing_in_list_raises(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', call_type=types.CallType.GENERATE_TEXT)}
    }
    with pytest.raises(ValueError, match='missing from models_by_call_type'):
      model_configs_instance._validate_models_by_call_type(configs, {})


class TestValidateModelsBySize:
  def test_valid_models_by_size(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', model_size_tags=[types.ModelSizeType.LARGE])}
    }
    by_size = {
        types.ModelSizeType.LARGE: [types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')]
    }
    model_configs_instance._validate_models_by_size(configs, by_size)

  def test_missing_in_list_raises(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4', model_size_tags=[types.ModelSizeType.LARGE])}
    }
    with pytest.raises(ValueError, match='missing from models_by_size'):
      model_configs_instance._validate_models_by_size(configs, {})

  def test_model_with_multiple_sizes_must_appear_in_all(self, model_configs_instance):
    """Model with multiple size tags must appear in all corresponding size lists."""
    pm = types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')
    configs = {
        'openai': {'gpt-4': create_provider_model_config(
            'openai', 'gpt-4',
            model_size_tags=[types.ModelSizeType.LARGE, types.ModelSizeType.LARGEST])}
    }
    by_size = {
        types.ModelSizeType.LARGE: [pm],
        types.ModelSizeType.LARGEST: [pm]
    }
    model_configs_instance._validate_models_by_size(configs, by_size)


class TestValidateDefaultModelPriorityList:
  def test_valid_priority_list(self, model_configs_instance):
    configs = {'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4')}}
    priority_list = [types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')]
    model_configs_instance._validate_default_model_priority_list(configs, priority_list)

  def test_invalid_provider_raises(self, model_configs_instance):
    configs = {'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4')}}
    priority_list = [types.ProviderModelType(provider='invalid', model='model', provider_model_identifier='id')]
    with pytest.raises(ValueError, match='not found'):
      model_configs_instance._validate_default_model_priority_list(configs, priority_list)


class TestValidateVersionConfig:
  def test_valid_complete_config(self, model_configs_instance):
    pm = types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id')
    configs = {
        'openai': {
            'gpt-4': create_provider_model_config(
                'openai', 'gpt-4', is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE])
        }
    }
    version_config = create_version_config(
        provider_model_configs=configs,
        featured_models={'openai': [pm]},
        models_by_call_type={types.CallType.GENERATE_TEXT: {'openai': [pm]}},
        models_by_size={types.ModelSizeType.LARGE: [pm]},
        default_model_priority_list=[pm])
    model_configs_instance._validate_version_config(version_config)

  def test_empty_configs(self, model_configs_instance):
    version_config = create_version_config(provider_model_configs=None)
    model_configs_instance._validate_version_config(version_config)


class TestValidateProviderModelConfigs:
  def test_valid_configs(self, model_configs_instance):
    configs = {
        'openai': {'gpt-4': create_provider_model_config('openai', 'gpt-4')},
        'claude': {'opus': create_provider_model_config('claude', 'opus')}
    }
    model_configs_instance._validate_provider_model_configs(configs)

  def test_provider_mismatch_raises(self, model_configs_instance):
    configs = {'openai': {'gpt-4': create_provider_model_config('wrong', 'gpt-4')}}
    with pytest.raises(ValueError, match='Provider mismatch'):
      model_configs_instance._validate_provider_model_configs(configs)


class TestValidateProviderModelConfig:
  def test_valid_config(self, model_configs_instance):
    config = create_provider_model_config('openai', 'gpt-4', model_size_tags=[types.ModelSizeType.LARGE])
    model_configs_instance._validate_provider_model_config('openai', 'gpt-4', config)

  def test_config_without_metadata(self, model_configs_instance):
    config = types.ProviderModelConfigType(
        provider_model=types.ProviderModelType(provider='openai', model='gpt-4', provider_model_identifier='id'),
        pricing=types.ProviderModelPricingType(per_query_token_cost=1.0, per_response_token_cost=1.0),
        metadata=None)
    model_configs_instance._validate_provider_model_config('openai', 'gpt-4', config)


# =============================================================================
# Get Methods Tests
# =============================================================================

class TestGetProviderModel:
  def test_with_tuple(self, model_configs_instance):
    result = model_configs_instance.get_provider_model(('openai', 'gpt-4o'))
    assert isinstance(result, types.ProviderModelType)
    assert result.provider == 'openai'

  def test_with_provider_model_type(self, model_configs_instance):
    pm = model_configs_instance.get_provider_model(('claude', 'opus-4'))
    result = model_configs_instance.get_provider_model(pm)
    assert result == pm


class TestGetProviderModelConfig:
  def test_returns_config(self, model_configs_instance):
    result = model_configs_instance.get_provider_model_config(('openai', 'gpt-4o'))
    assert isinstance(result, types.ProviderModelConfigType)
    assert result.provider_model.provider == 'openai'


class TestGetProviderModelCost:
  def test_calculates_cost(self, model_configs_instance):
    cost = model_configs_instance.get_provider_model_cost(('openai', 'gpt-4o'), 100, 200)
    assert isinstance(cost, int)
    assert cost >= 0

  def test_zero_tokens(self, model_configs_instance):
    cost = model_configs_instance.get_provider_model_cost(('openai', 'gpt-4o'), 0, 0)
    assert cost == 0


# =============================================================================
# GetAllModels Filter Tests
# =============================================================================

class TestGetAllModels:
  @pytest.fixture
  def custom_configs(self):
    """Create ModelConfigs with controlled test data."""
    pm_featured = types.ProviderModelType(provider='test', model='featured', provider_model_identifier='id1')
    pm_not_featured = types.ProviderModelType(provider='test', model='not_featured', provider_model_identifier='id2')

    configs = {
        'test': {
            'featured': create_provider_model_config(
                'test', 'featured', is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE]),
            'not_featured': create_provider_model_config(
                'test', 'not_featured', is_featured=False,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.SMALL]),
        }
    }
    version_config = create_version_config(
        provider_model_configs=configs,
        featured_models={'test': [pm_featured]},
        models_by_call_type={
            types.CallType.GENERATE_TEXT: {'test': [pm_featured, pm_not_featured]}
        },
        models_by_size={
            types.ModelSizeType.LARGE: [pm_featured],
            types.ModelSizeType.SMALL: [pm_not_featured]
        },
        default_model_priority_list=[pm_featured])
    schema = create_schema(version_config)
    model_configs_params = model_configs.ModelConfigsParams(
        model_configs_schema=schema)
    return model_configs.ModelConfigs(init_from_params=model_configs_params)

  def test_only_featured_filter(self, custom_configs):
    result = custom_configs.get_all_models(call_type=None, only_featured=True)
    model_names = [m.model for m in result]
    assert 'featured' in model_names
    assert 'not_featured' not in model_names

  def test_model_size_filter(self, custom_configs):
    result = custom_configs.get_all_models(model_size=types.ModelSizeType.LARGE, call_type=None, only_featured=False)
    model_names = [m.model for m in result]
    assert 'featured' in model_names
    assert 'not_featured' not in model_names

  def test_invalid_provider_raises(self, custom_configs):
    with pytest.raises(ValueError, match='Provider not supported'):
      custom_configs.get_all_models(provider='invalid')


# =============================================================================
# Load Config Tests
# =============================================================================

class TestLoadModelConfigFromJsonString:
  def test_load_example_proxdash_config(self, model_configs_instance):
    """Test loading the example proxdash config file."""
    from importlib import resources
    config_data = (
        resources.files("proxai.connectors.model_configs_data")
        .joinpath("example_proxdash_model_configs.json")
        .read_text(encoding="utf-8")
    )
    model_configs_instance.load_model_config_from_json_string(config_data)

    provider_configs = model_configs_instance.model_configs_schema.version_config.provider_model_configs
    assert 'openai' in provider_configs
    assert 'gemini' in provider_configs


# =============================================================================
# Built-in Config Validation Tests
# =============================================================================

class TestBuiltInConfigValidation:
  def test_built_in_config_loads(self):
    mc = model_configs.ModelConfigs()
    assert mc.model_configs_schema is not None

  def test_provider_model_configs_valid(self, model_configs_instance):
    configs = model_configs_instance.model_configs_schema.version_config.provider_model_configs
    model_configs_instance._validate_provider_model_configs(configs)

  def test_featured_models_consistent(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    model_configs_instance._validate_featured_models(
        schema.version_config.provider_model_configs,
        schema.version_config.featured_models)

  def test_models_by_call_type_consistent(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    model_configs_instance._validate_models_by_call_type(
        schema.version_config.provider_model_configs,
        schema.version_config.models_by_call_type)

  def test_models_by_size_consistent(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    model_configs_instance._validate_models_by_size(
        schema.version_config.provider_model_configs,
        schema.version_config.models_by_size)

import copy
import pytest
import proxai.types as types
import proxai.connectors.model_configs as model_configs


def create_minimal_provider_model_config(
    provider: str,
    model: str,
    is_featured: bool = True,
    call_type: types.CallType = types.CallType.GENERATE_TEXT,
    model_size_tags: list = None
) -> types.ProviderModelConfigType:
  """Helper to create a minimal ProviderModelConfigType."""
  return types.ProviderModelConfigType(
      provider_model=types.ProviderModelType(
          provider=provider,
          model=model,
          provider_model_identifier=f'{model}-identifier'),
      pricing=types.ProviderModelPricingType(
          per_response_token_cost=1.0,
          per_query_token_cost=1.0),
      features=types.ProviderModelFeatureType(),
      metadata=types.ProviderModelMetadataType(
          call_type=call_type,
          is_featured=is_featured,
          model_size_tags=model_size_tags))


def create_minimal_version_config(
    provider_model_configs: dict,
    featured_models: dict = None,
    models_by_call_type: dict = None,
    models_by_size: dict = None,
    default_model_priority_list: list = None
) -> types.ModelConfigsSchemaVersionConfigType:
  """Helper to create a minimal ModelConfigsSchemaVersionConfigType."""
  return types.ModelConfigsSchemaVersionConfigType(
      provider_model_configs=provider_model_configs,
      featured_models=featured_models or {},
      models_by_call_type=models_by_call_type or {},
      models_by_size=models_by_size or {},
      default_model_priority_list=default_model_priority_list or [])


def create_minimal_schema(
    version_config: types.ModelConfigsSchemaVersionConfigType
) -> types.ModelConfigsSchemaType:
  """Helper to create a minimal ModelConfigsSchemaType."""
  return types.ModelConfigsSchemaType(
      metadata=types.ModelConfigsSchemaMetadataType(version='1.0.0'),
      version_config=version_config)


@pytest.fixture
def model_configs_instance():
  """Fixture to provide a ModelConfigs instance for testing."""
  return model_configs.ModelConfigs()


class TestIsProviderModelTuple:
  def test_valid_tuple(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(
        ('openai', 'gpt-4')) is True

  def test_invalid_tuple_length(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(
        ('openai',)) is False
    assert model_configs_instance._is_provider_model_tuple(
        ('openai', 'gpt-4', 'extra')) is False

  def test_invalid_tuple_types(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(
        (1, 'gpt-4')) is False
    assert model_configs_instance._is_provider_model_tuple(
        ('openai', 123)) is False

  def test_not_tuple(self, model_configs_instance):
    assert model_configs_instance._is_provider_model_tuple(
        ['openai', 'gpt-4']) is False
    assert model_configs_instance._is_provider_model_tuple(
        'openai') is False


class TestCheckProviderModelIdentifierType:
  def test_not_supported_provider(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='not_supported_provider',
              model='not_supported_model',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_not_supported_model(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='openai',
              model='not_supported_model',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_not_supported_provider_model_identifier(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='claude',
              model='opus-4',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_supported_provider_model_identifier(self, model_configs_instance):
    provider_model = model_configs_instance.get_provider_model(
        ('claude', 'opus-4'))
    model_configs_instance.check_provider_model_identifier_type(provider_model)

  def test_not_supported_provider_tuple(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          ('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model_tuple(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance.check_provider_model_identifier_type(
          ('openai', 'not_supported_model'))

  def test_supported_model_tuple(self, model_configs_instance):
    model_configs_instance.check_provider_model_identifier_type(
        ('claude', 'opus-4'))


class TestGetProviderModelKey:
  def test_with_provider_model_type(self, model_configs_instance):
    pm = types.ProviderModelType(
        provider='openai', model='gpt-4', provider_model_identifier='gpt-4-id')
    result = model_configs_instance._get_provider_model_key(pm)
    assert result == ('openai', 'gpt-4')

  def test_with_tuple(self, model_configs_instance):
    result = model_configs_instance._get_provider_model_key(('claude', 'opus'))
    assert result == ('claude', 'opus')

  def test_with_invalid_type(self, model_configs_instance):
    with pytest.raises(ValueError):
      model_configs_instance._get_provider_model_key('invalid')


class TestGetAllFeaturedModelsFromConfigs:
  def test_returns_featured_models(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', is_featured=True),
            'gpt-3': create_minimal_provider_model_config(
                'openai', 'gpt-3', is_featured=False),
        },
        'claude': {
            'opus': create_minimal_provider_model_config(
                'claude', 'opus', is_featured=True),
        }
    }
    result = model_configs_instance._get_all_featured_models_from_configs(
        configs)
    assert result == {('openai', 'gpt-4'), ('claude', 'opus')}

  def test_empty_configs(self, model_configs_instance):
    result = model_configs_instance._get_all_featured_models_from_configs({})
    assert result == set()


class TestGetAllModelsByCallTypeFromConfigs:
  def test_groups_by_call_type(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                call_type=types.CallType.GENERATE_TEXT),
        },
        'claude': {
            'opus': create_minimal_provider_model_config(
                'claude', 'opus',
                call_type=types.CallType.GENERATE_TEXT),
        }
    }
    result = model_configs_instance._get_all_models_by_call_type_from_configs(
        configs)
    assert types.CallType.GENERATE_TEXT in result
    assert result[types.CallType.GENERATE_TEXT] == {
        ('openai', 'gpt-4'), ('claude', 'opus')}

  def test_model_without_call_type(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', call_type=None),
        }
    }
    result = model_configs_instance._get_all_models_by_call_type_from_configs(
        configs)
    assert result == {}


class TestGetAllModelsBySizeFromConfigs:
  def test_groups_by_size(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[types.ModelSizeType.LARGE]),
            'gpt-mini': create_minimal_provider_model_config(
                'openai', 'gpt-mini',
                model_size_tags=[types.ModelSizeType.SMALL]),
        }
    }
    result = model_configs_instance._get_all_models_by_size_from_configs(
        configs)
    assert result[types.ModelSizeType.LARGE] == {('openai', 'gpt-4')}
    assert result[types.ModelSizeType.SMALL] == {('openai', 'gpt-mini')}

  def test_model_with_multiple_size_tags(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[
                    types.ModelSizeType.LARGE,
                    types.ModelSizeType.LARGEST]),
        }
    }
    result = model_configs_instance._get_all_models_by_size_from_configs(
        configs)
    assert ('openai', 'gpt-4') in result[types.ModelSizeType.LARGE]
    assert ('openai', 'gpt-4') in result[types.ModelSizeType.LARGEST]

  def test_model_without_size_tags(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', model_size_tags=None),
        }
    }
    result = model_configs_instance._get_all_models_by_size_from_configs(
        configs)
    assert result == {}


class TestValidateFeaturedModels:
  def test_valid_featured_models(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', is_featured=True),
        }
    }
    featured = {
        'openai': [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ]
    }
    model_configs_instance._validate_featured_models(configs, featured)

  def test_missing_in_featured_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', is_featured=True),
        }
    }
    featured = {}
    with pytest.raises(ValueError, match='missing from featured_models'):
      model_configs_instance._validate_featured_models(configs, featured)

  def test_extra_in_featured_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', is_featured=False),
        }
    }
    featured = {
        'openai': [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ]
    }
    with pytest.raises(ValueError, match='not marked as is_featured=True'):
      model_configs_instance._validate_featured_models(configs, featured)


class TestValidateModelsByCallType:
  def test_valid_models_by_call_type(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                call_type=types.CallType.GENERATE_TEXT),
        }
    }
    by_call_type = {
        types.CallType.GENERATE_TEXT: {
            'openai': [
                types.ProviderModelType(
                    provider='openai', model='gpt-4',
                    provider_model_identifier='gpt-4-id')
            ]
        }
    }
    model_configs_instance._validate_models_by_call_type(
        configs, by_call_type)

  def test_missing_in_call_type_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                call_type=types.CallType.GENERATE_TEXT),
        }
    }
    by_call_type = {}
    with pytest.raises(ValueError, match='missing from models_by_call_type'):
      model_configs_instance._validate_models_by_call_type(
          configs, by_call_type)

  def test_extra_in_call_type_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', call_type=None),
        }
    }
    by_call_type = {
        types.CallType.GENERATE_TEXT: {
            'openai': [
                types.ProviderModelType(
                    provider='openai', model='gpt-4',
                    provider_model_identifier='gpt-4-id')
            ]
        }
    }
    with pytest.raises(ValueError, match='not marked with that call_type'):
      model_configs_instance._validate_models_by_call_type(
          configs, by_call_type)


class TestValidateModelsBySize:
  def test_valid_models_by_size(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }
    by_size = {
        types.ModelSizeType.LARGE: [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ]
    }
    model_configs_instance._validate_models_by_size(configs, by_size)

  def test_missing_in_size_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }
    by_size = {}
    with pytest.raises(ValueError, match='missing from models_by_size'):
      model_configs_instance._validate_models_by_size(configs, by_size)

  def test_extra_in_size_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4', model_size_tags=None),
        }
    }
    by_size = {
        types.ModelSizeType.LARGE: [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ]
    }
    with pytest.raises(ValueError, match='does not contain'):
      model_configs_instance._validate_models_by_size(configs, by_size)

  def test_model_with_multiple_sizes(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                model_size_tags=[
                    types.ModelSizeType.LARGE,
                    types.ModelSizeType.LARGEST]),
        }
    }
    by_size = {
        types.ModelSizeType.LARGE: [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ],
        types.ModelSizeType.LARGEST: [
            types.ProviderModelType(
                provider='openai', model='gpt-4',
                provider_model_identifier='gpt-4-id')
        ]
    }
    model_configs_instance._validate_models_by_size(configs, by_size)


class TestValidateDefaultModelPriorityList:
  def test_valid_priority_list(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'gpt-4'),
        }
    }
    priority_list = [
        types.ProviderModelType(
            provider='openai', model='gpt-4',
            provider_model_identifier='gpt-4-id')
    ]
    model_configs_instance._validate_default_model_priority_list(
        configs, priority_list)

  def test_invalid_provider(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'gpt-4'),
        }
    }
    priority_list = [
        types.ProviderModelType(
            provider='invalid', model='model',
            provider_model_identifier='id')
    ]
    with pytest.raises(ValueError, match='not found in provider_model_configs'):
      model_configs_instance._validate_default_model_priority_list(
          configs, priority_list)

  def test_invalid_model(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'gpt-4'),
        }
    }
    priority_list = [
        types.ProviderModelType(
            provider='openai', model='invalid',
            provider_model_identifier='id')
    ]
    with pytest.raises(ValueError, match='not found in provider_model_configs'):
      model_configs_instance._validate_default_model_priority_list(
          configs, priority_list)

  def test_with_tuple_format(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'gpt-4'),
        }
    }
    priority_list = [('openai', 'gpt-4')]
    model_configs_instance._validate_default_model_priority_list(
        configs, priority_list)


class TestValidateVersionConfig:
  def test_valid_complete_config(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }
    pm = types.ProviderModelType(
        provider='openai', model='gpt-4', provider_model_identifier='gpt-4-id')
    version_config = create_minimal_version_config(
        provider_model_configs=configs,
        featured_models={'openai': [pm]},
        models_by_call_type={
            types.CallType.GENERATE_TEXT: {'openai': [pm]}},
        models_by_size={types.ModelSizeType.LARGE: [pm]},
        default_model_priority_list=[pm])
    model_configs_instance._validate_version_config(version_config)

  def test_empty_provider_model_configs(self, model_configs_instance):
    version_config = create_minimal_version_config(
        provider_model_configs=None)
    model_configs_instance._validate_version_config(version_config)

  def test_skips_none_fields(self, model_configs_instance):
    """Test that None fields are skipped without validation errors."""
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config(
                'openai', 'gpt-4',
                is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }
    version_config = types.ModelConfigsSchemaVersionConfigType(
        provider_model_configs=configs,
        featured_models=None,
        models_by_call_type=None,
        models_by_size=None,
        default_model_priority_list=None)
    model_configs_instance._validate_version_config(version_config)


class TestValidateProviderModelKeyMatchesConfig:
  def test_valid_config(self, model_configs_instance):
    config = create_minimal_provider_model_config('openai', 'gpt-4')
    model_configs_instance._validate_provider_model_key_matches_config(
        'openai', 'gpt-4', config)

  def test_provider_model_is_none(self, model_configs_instance):
    config = types.ProviderModelConfigType(provider_model=None)
    with pytest.raises(ValueError, match='provider_model is None'):
      model_configs_instance._validate_provider_model_key_matches_config(
          'openai', 'gpt-4', config)

  def test_provider_mismatch(self, model_configs_instance):
    config = create_minimal_provider_model_config('claude', 'gpt-4')
    with pytest.raises(ValueError, match='Provider mismatch'):
      model_configs_instance._validate_provider_model_key_matches_config(
          'openai', 'gpt-4', config)

  def test_model_mismatch(self, model_configs_instance):
    config = create_minimal_provider_model_config('openai', 'gpt-3')
    with pytest.raises(ValueError, match='Model mismatch'):
      model_configs_instance._validate_provider_model_key_matches_config(
          'openai', 'gpt-4', config)


class TestValidatePricing:
  def test_valid_pricing(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(
        per_query_token_cost=1.0,
        per_response_token_cost=2.0)
    model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)

  def test_zero_pricing_is_valid(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(
        per_query_token_cost=0.0,
        per_response_token_cost=0.0)
    model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)

  def test_pricing_is_none(self, model_configs_instance):
    with pytest.raises(ValueError, match='pricing is None'):
      model_configs_instance._validate_pricing('openai', 'gpt-4', None)

  def test_negative_query_cost(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(
        per_query_token_cost=-1.0,
        per_response_token_cost=2.0)
    with pytest.raises(ValueError, match='per_query_token_cost is negative'):
      model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)

  def test_negative_response_cost(self, model_configs_instance):
    pricing = types.ProviderModelPricingType(
        per_query_token_cost=1.0,
        per_response_token_cost=-2.0)
    with pytest.raises(ValueError, match='per_response_token_cost is negative'):
      model_configs_instance._validate_pricing('openai', 'gpt-4', pricing)


class TestValidateModelSizeTags:
  def test_valid_size_tags(self, model_configs_instance):
    model_configs_instance._validate_model_size_tags(
        'openai', 'gpt-4',
        [types.ModelSizeType.LARGE, types.ModelSizeType.LARGEST])

  def test_empty_size_tags(self, model_configs_instance):
    model_configs_instance._validate_model_size_tags('openai', 'gpt-4', [])

  def test_invalid_size_tag_string(self, model_configs_instance):
    with pytest.raises(ValueError, match='Invalid model_size_tag'):
      model_configs_instance._validate_model_size_tags(
          'openai', 'gpt-4', ['invalid_size'])

  def test_invalid_size_tag_mixed(self, model_configs_instance):
    with pytest.raises(ValueError, match='Invalid model_size_tag'):
      model_configs_instance._validate_model_size_tags(
          'openai', 'gpt-4', [types.ModelSizeType.LARGE, 'invalid'])


class TestValidateProviderModelConfig:
  def test_valid_config(self, model_configs_instance):
    config = create_minimal_provider_model_config(
        'openai', 'gpt-4',
        model_size_tags=[types.ModelSizeType.LARGE])
    model_configs_instance._validate_provider_model_config(
        'openai', 'gpt-4', config)

  def test_config_without_metadata(self, model_configs_instance):
    config = types.ProviderModelConfigType(
        provider_model=types.ProviderModelType(
            provider='openai', model='gpt-4',
            provider_model_identifier='gpt-4-id'),
        pricing=types.ProviderModelPricingType(
            per_query_token_cost=1.0,
            per_response_token_cost=1.0),
        metadata=None)
    model_configs_instance._validate_provider_model_config(
        'openai', 'gpt-4', config)

  def test_config_with_none_size_tags(self, model_configs_instance):
    config = create_minimal_provider_model_config(
        'openai', 'gpt-4', model_size_tags=None)
    model_configs_instance._validate_provider_model_config(
        'openai', 'gpt-4', config)


class TestValidateProviderModelConfigs:
  def test_valid_configs(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'gpt-4'),
        },
        'claude': {
            'opus': create_minimal_provider_model_config('claude', 'opus'),
        }
    }
    model_configs_instance._validate_provider_model_configs(configs)

  def test_empty_configs(self, model_configs_instance):
    model_configs_instance._validate_provider_model_configs({})

  def test_catches_provider_mismatch(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('wrong_provider', 'gpt-4'),
        }
    }
    with pytest.raises(ValueError, match='Provider mismatch'):
      model_configs_instance._validate_provider_model_configs(configs)

  def test_catches_model_mismatch(self, model_configs_instance):
    configs = {
        'openai': {
            'gpt-4': create_minimal_provider_model_config('openai', 'wrong_model'),
        }
    }
    with pytest.raises(ValueError, match='Model mismatch'):
      model_configs_instance._validate_provider_model_configs(configs)

  def test_catches_negative_pricing(self, model_configs_instance):
    config = create_minimal_provider_model_config('openai', 'gpt-4')
    config.pricing.per_query_token_cost = -1.0
    configs = {'openai': {'gpt-4': config}}
    with pytest.raises(ValueError, match='per_query_token_cost is negative'):
      model_configs_instance._validate_provider_model_configs(configs)


class TestBuiltInConfigValidation:
  """Test that the built-in v1.0.0.json config passes validation."""

  def test_built_in_config_passes_validation(self):
    mc = model_configs.ModelConfigs()
    assert mc.model_configs_schema is not None

  def test_provider_model_configs_validation(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    configs = schema.version_config.provider_model_configs
    model_configs_instance._validate_provider_model_configs(configs)

  def test_featured_models_consistency(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    configs = schema.version_config.provider_model_configs
    featured = schema.version_config.featured_models
    model_configs_instance._validate_featured_models(configs, featured)

  def test_models_by_call_type_consistency(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    configs = schema.version_config.provider_model_configs
    by_call_type = schema.version_config.models_by_call_type
    model_configs_instance._validate_models_by_call_type(configs, by_call_type)

  def test_models_by_size_consistency(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    configs = schema.version_config.provider_model_configs
    by_size = schema.version_config.models_by_size
    model_configs_instance._validate_models_by_size(configs, by_size)

  def test_default_model_priority_list_consistency(self, model_configs_instance):
    schema = model_configs_instance.model_configs_schema
    configs = schema.version_config.provider_model_configs
    priority_list = schema.version_config.default_model_priority_list
    model_configs_instance._validate_default_model_priority_list(
        configs, priority_list)

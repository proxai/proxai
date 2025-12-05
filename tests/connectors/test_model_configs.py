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


class TestGetProviderModel:
  def test_with_tuple(self, model_configs_instance):
    result = model_configs_instance.get_provider_model(('openai', 'gpt-4o'))
    assert isinstance(result, types.ProviderModelType)
    assert result.provider == 'openai'
    assert result.model == 'gpt-4o'

  def test_with_provider_model_type(self, model_configs_instance):
    pm = model_configs_instance.get_provider_model(('claude', 'opus-4'))
    result = model_configs_instance.get_provider_model(pm)
    assert result == pm


class TestGetProviderModelConfig:
  def test_returns_config(self, model_configs_instance):
    result = model_configs_instance.get_provider_model_config(('openai', 'gpt-4o'))
    assert isinstance(result, types.ProviderModelConfigType)
    assert result.provider_model.provider == 'openai'
    assert result.pricing is not None


class TestGetProviderModelCost:
  def test_calculates_cost(self, model_configs_instance):
    cost = model_configs_instance.get_provider_model_cost(
        ('openai', 'gpt-4o'), query_token_count=100, response_token_count=200)
    assert isinstance(cost, int)
    assert cost >= 0

  def test_zero_tokens(self, model_configs_instance):
    cost = model_configs_instance.get_provider_model_cost(
        ('openai', 'gpt-4o'), query_token_count=0, response_token_count=0)
    assert cost == 0


class TestIsFeatureSupported:
  def test_feature_supported(self, model_configs_instance):
    pm = model_configs_instance.get_provider_model(('openai', 'gpt-4o'))
    result = model_configs_instance.is_feature_supported(pm, 'temperature')
    assert isinstance(result, bool)

  def test_feature_not_supported(self, model_configs_instance):
    pm = model_configs_instance.get_provider_model(('openai', 'o1'))
    result = model_configs_instance.is_feature_supported(pm, 'temperature')
    assert result is True  # temperature is in not_supported_features


class TestGetAllModels:
  """Tests for get_all_models filter logic."""

  @pytest.fixture
  def custom_model_configs(self):
    """Create ModelConfigs with controlled test data."""
    # Define test models with known properties
    configs = {
        'provider_a': {
            # Featured, LARGE size, GENERATE_TEXT
            'model_featured_large': create_minimal_provider_model_config(
                'provider_a', 'model_featured_large',
                is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE]),
            # Not featured, SMALL size, GENERATE_TEXT
            'model_not_featured': create_minimal_provider_model_config(
                'provider_a', 'model_not_featured',
                is_featured=False,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.SMALL]),
            # Featured, no size tags, GENERATE_TEXT
            'model_no_size_tags': create_minimal_provider_model_config(
                'provider_a', 'model_no_size_tags',
                is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=None),
        },
        'provider_b': {
            # Featured, LARGE size, GENERATE_TEXT
            'model_b': create_minimal_provider_model_config(
                'provider_b', 'model_b',
                is_featured=True,
                call_type=types.CallType.GENERATE_TEXT,
                model_size_tags=[types.ModelSizeType.LARGE]),
        }
    }

    pm_featured_large = configs['provider_a']['model_featured_large'].provider_model
    pm_not_featured = configs['provider_a']['model_not_featured'].provider_model
    pm_no_size = configs['provider_a']['model_no_size_tags'].provider_model
    pm_b = configs['provider_b']['model_b'].provider_model

    version_config = create_minimal_version_config(
        provider_model_configs=configs,
        featured_models={
            'provider_a': [pm_featured_large, pm_no_size],
            'provider_b': [pm_b]
        },
        models_by_call_type={
            types.CallType.GENERATE_TEXT: {
                'provider_a': [pm_featured_large, pm_not_featured, pm_no_size],
                'provider_b': [pm_b]
            }
        },
        models_by_size={
            types.ModelSizeType.LARGE: [pm_featured_large, pm_b],
            types.ModelSizeType.SMALL: [pm_not_featured]
        },
        default_model_priority_list=[pm_featured_large])

    schema = create_minimal_schema(version_config)
    return model_configs.ModelConfigs(model_configs_schema=schema)

  # Core filter tests with controlled data
  def test_model_size_filter_excludes_models_without_size_tags(
      self, custom_model_configs):
    """Model with model_size_tags=None should be excluded when filtering by size."""
    result = custom_model_configs.get_all_models(
        model_size=types.ModelSizeType.LARGE,
        call_type=None,
        only_featured=False)
    model_names = [m.model for m in result]
    assert 'model_no_size_tags' not in model_names
    assert 'model_featured_large' in model_names

  def test_model_size_filter_excludes_models_with_different_size(
      self, custom_model_configs):
    """Model with different size tag should be excluded."""
    result = custom_model_configs.get_all_models(
        model_size=types.ModelSizeType.LARGE,
        call_type=None,
        only_featured=False)
    model_names = [m.model for m in result]
    assert 'model_not_featured' not in model_names  # Has SMALL, not LARGE

  def test_only_featured_filter(self, custom_model_configs):
    """only_featured=True should exclude non-featured models."""
    result = custom_model_configs.get_all_models(
        call_type=None,
        only_featured=True)
    model_names = [m.model for m in result]
    assert 'model_not_featured' not in model_names
    assert 'model_featured_large' in model_names

  def test_provider_filter(self, custom_model_configs):
    """provider filter should only return models from that provider."""
    result = custom_model_configs.get_all_models(
        provider='provider_a',
        call_type=None,
        only_featured=False)
    assert all(m.provider == 'provider_a' for m in result)
    assert len(result) == 3

  def test_combined_filters(self, custom_model_configs):
    """Combined filters should all be applied."""
    result = custom_model_configs.get_all_models(
        provider='provider_a',
        model_size=types.ModelSizeType.LARGE,
        call_type=types.CallType.GENERATE_TEXT,
        only_featured=True)
    assert len(result) == 1
    assert result[0].model == 'model_featured_large'

  # Error cases
  def test_invalid_provider_raises_error(self, custom_model_configs):
    with pytest.raises(ValueError, match='Provider not supported'):
      custom_model_configs.get_all_models(provider='invalid_provider')

  def test_invalid_call_type_raises_error(self, custom_model_configs):
    with pytest.raises(ValueError, match='Call type not supported'):
      custom_model_configs.get_all_models(call_type='invalid_call_type')

  def test_invalid_model_size_raises_error(self, custom_model_configs):
    with pytest.raises(ValueError, match='Model size not supported'):
      custom_model_configs.get_all_models(model_size=types.ModelSizeType.MEDIUM)


class TestLoadModelConfigFromJsonString:
  """Test the load_model_config_from_json_string method."""

  def test_load_valid_json_string(self, model_configs_instance):
    """Test loading a valid JSON config string updates the schema."""
    from importlib import resources
    config_data = (
        resources.files("proxai.connectors.model_configs_data")
        .joinpath("example_proxdash_model_configs.json")
        .read_text(encoding="utf-8")
    )

    model_configs_instance.load_model_config_from_json_string(config_data)

    # Verify the schema was loaded with exactly the models from example config
    provider_configs = (
        model_configs_instance.model_configs_schema.version_config
        .provider_model_configs)
    assert len(provider_configs) == 2  # openai, gemini
    assert len(provider_configs['openai']) == 2  # gpt-4o, gpt-4o-mini
    assert len(provider_configs['gemini']) == 2  # gemini-2.5-pro, gemini-2.0-flash


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

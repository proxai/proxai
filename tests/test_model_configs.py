import pytest
import proxai.types as types
import proxai.connectors.model_configs as model_configs


@pytest.fixture
def model_configs_instance():
  """Fixture to provide a ModelConfigs instance for testing."""
  return model_configs.ModelConfigs()


class TestIsProviderModelTuple:
  def test_valid_tuple(self, model_configs_instance):
    assert model_configs_instance.is_provider_model_tuple(
        ('openai', 'gpt-4')) is True

  def test_invalid_tuple_length(self, model_configs_instance):
    assert model_configs_instance.is_provider_model_tuple(
        ('openai',)) is False
    assert model_configs_instance.is_provider_model_tuple(
        ('openai', 'gpt-4', 'extra')) is False

  def test_invalid_tuple_types(self, model_configs_instance):
    assert model_configs_instance.is_provider_model_tuple(
        (1, 'gpt-4')) is False
    assert model_configs_instance.is_provider_model_tuple(
        ('openai', 123)) is False

  def test_not_tuple(self, model_configs_instance):
    assert model_configs_instance.is_provider_model_tuple(
        ['openai', 'gpt-4']) is False
    assert model_configs_instance.is_provider_model_tuple(
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
    provider_model = model_configs_instance.get_provider_model_config(
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

  def test_print_model_configs_schema(self, model_configs_instance):
    model_configs_instance.print_model_configs_schema()

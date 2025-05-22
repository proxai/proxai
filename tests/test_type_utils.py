import datetime
import proxai.types as types
import proxai.type_utils as type_utils
import proxai.connectors.model_configs as model_configs
import pytest


class TestCheckProviderModelType:
  def test_not_supported_provider(self):
    with pytest.raises(ValueError):
      type_utils.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='not_supported_provider',
              model='not_supported_model',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      type_utils.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='openai',
              model='not_supported_model',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_not_supported_provider_model_identifier(self):
    with pytest.raises(ValueError):
      type_utils.check_provider_model_identifier_type(
          types.ProviderModelType(
              provider='claude',
              model='opus-4',
              provider_model_identifier=(
                  'not_supported_provider_model_identifier')))

  def test_supported_provider_model_identifier(self):
    type_utils.check_provider_model_identifier_type(
        model_configs.ALL_MODELS['claude']['opus-4'])

  def test_not_supported_provider_tuple(self):
    with pytest.raises(ValueError):
      type_utils.check_provider_model_identifier_type(
          ('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model_tuple(self):
    with pytest.raises(ValueError):
      type_utils.check_provider_model_identifier_type(
          ('openai', 'not_supported_model'))

  def test_supported_model_tuple(self):
    type_utils.check_provider_model_identifier_type(
        ('claude', 'opus-4'))


class TestCheckMessagesType:
  def test_invalid_message_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type(['invalid_message'])

  def test_invalid_message_keys(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 'user'}])

  def test_invalid_role_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 1, 'content': 'content'}])

  def test_invalid_content_type(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([{'role': 'user', 'content': 1}])

  def test_invalid_role(self):
    with pytest.raises(ValueError):
      type_utils.check_messages_type([
          {'role': 'invalid_role', 'content': 'content'}])

  def test_valid_message(self):
    type_utils.check_messages_type([
      {'role': 'user', 'content': 'content'}])

import datetime
import proxai.type_utils as type_utils
import pytest


class TestCheckModelType:
  def test_not_supported_provider(self):
    with pytest.raises(ValueError):
      type_utils.check_model_type(
          ('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      type_utils.check_model_type(('openai', 'not_supported_model'))

  def test_supported_model(self):
    type_utils.check_model_type(('claude', 'claude-3-opus-20240229'))


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

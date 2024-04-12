import datetime
import proxai.type_utils as type_utils
import pytest


class TestDatetime:
  def test_encode_decode(self):
    dt = datetime.datetime.now()
    dt_str = type_utils.encode_datetime(dt)
    assert type_utils.decode_datetime(dt_str) == dt


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

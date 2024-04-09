import proxai.types as types
from proxai import proxai
import pytest


class TestRunType:
    def test_setup_run_type(self):
      proxai._set_run_type(types.RunType.TEST)
      assert proxai._RUN_TYPE == types.RunType.TEST


class TestRegisterModel:
  def test_not_supported_provider(self):
    with pytest.raises(ValueError):
      proxai.set_model(
          generate_text=('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      proxai.set_model(generate_text=('openai', 'not_supported_model'))

  def test_successful_register_model(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=('openai', 'gpt-3.5-turbo'))
    assert proxai._REGISTERED_VALUES['generate_text'] == (
        'openai', 'gpt-3.5-turbo')


class TestGenerateText:
  def _test_generate_text(self, model: types.ModelType):
    proxai._set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=model)
    assert proxai._REGISTERED_VALUES['generate_text'] == model

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'
    assert model in proxai._INITIALIZED_MODEL_CONNECTORS
    assert proxai._INITIALIZED_MODEL_CONNECTORS[model] is not None

  def test_openai(self):
    self._test_generate_text(('openai', 'gpt-3.5-turbo'))

  def test_claude(self):
    self._test_generate_text(('claude', 'claude-3-opus-20240229'))

  def test_gemini(self):
    self._test_generate_text(('gemini', 'models/gemini-1.0-pro'))

  def test_cohere(self):
    self._test_generate_text(('cohere', 'command-r'))

  def test_databricks(self):
    self._test_generate_text(('databricks', 'databricks-dbrx-instruct'))

  def test_mistral(self):
    self._test_generate_text(('mistral', 'open-mistral-7b'))

  def test_hugging_face(self):
    self._test_generate_text(('hugging_face', 'google/gemma-7b-it'))

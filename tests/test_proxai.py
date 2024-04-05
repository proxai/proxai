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
      proxai.register_model('not_supported_provider', 'not_supported_model')

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      proxai.register_model('openai', 'not_supported_model')

  def test_not_implemented_provider(self):
    with pytest.raises(ValueError):
      proxai.register_model('hugging_face', 'not_implemented_model')

  def test_successful_register_model(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('openai', 'gpt-3.5-turbo')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.OPENAI
    assert proxai._REGISTERED_SIGNATURE.model == types.OpenAIModel.GPT_3_5_TURBO
    assert proxai._MODEL_CONNECTOR is not None
    assert proxai._MODEL_CONNECTOR.model_signature == proxai._REGISTERED_SIGNATURE
    assert proxai._MODEL_CONNECTOR.run_type == types.RunType.TEST


class TestGenerateText:
  def test_openai(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('openai', 'gpt-4')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.OPENAI
    assert proxai._REGISTERED_SIGNATURE.model == types.OpenAIModel.GPT_4
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.OpenAIModel.GPT_4

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_claude(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('claude', 'claude-3-opus-20240229')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.CLAUDE
    assert proxai._REGISTERED_SIGNATURE.model == types.ClaudeModel.CLAUDE_3_OPUS
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.ClaudeModel.CLAUDE_3_OPUS

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_gemini(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('gemini', 'models/gemini-pro')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.GEMINI
    assert proxai._REGISTERED_SIGNATURE.model == types.GeminiModel.GEMINI_PRO
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.GeminiModel.GEMINI_PRO

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_cohere(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('cohere', 'command-r')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.COHERE
    assert proxai._REGISTERED_SIGNATURE.model == types.CohereModel.COMMAND_R
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.CohereModel.COMMAND_R

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_databricks(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('databricks', 'databricks-dbrx-instruct')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.DATABRICKS
    assert proxai._REGISTERED_SIGNATURE.model == types.DatabricksModel.DATABRICKS_DBRX_INSTRUCT
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.DatabricksModel.DATABRICKS_DBRX_INSTRUCT

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_mistral(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('mistral', 'open-mistral-7b')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.MISTRAL
    assert proxai._REGISTERED_SIGNATURE.model == types.MistralModel.OPEN_MISTRAL_7B
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.MistralModel.OPEN_MISTRAL_7B

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

  def test_hugging_face(self):
    proxai._set_run_type(types.RunType.TEST)
    proxai.register_model('hugging_face', 'google/gemma-7b-it')
    assert proxai._REGISTERED_SIGNATURE.provider == types.Provider.HUGGING_FACE
    assert proxai._REGISTERED_SIGNATURE.model == types.HuggingFaceModel.GOOGLE_GEMMA_7B_IT
    assert proxai._MODEL_CONNECTOR.model_signature.model == types.HuggingFaceModel.GOOGLE_GEMMA_7B_IT

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'

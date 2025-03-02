import pytest
import proxai.types as types
import proxai.connectors.model_configs as model_configs


class TestModelConfigs:
  def test_all_models(self):
    assert model_configs.ALL_MODELS is not None
    for provider in model_configs.ALL_MODELS:
      for model in model_configs.ALL_MODELS[provider]:
        assert isinstance(
            model_configs.ALL_MODELS[provider][model], types.ProviderModelType)
        assert model_configs.ALL_MODELS[provider][model].provider == provider
        assert model_configs.ALL_MODELS[provider][model].model == model

  def test_generate_text_models(self):
    assert model_configs.GENERATE_TEXT_MODELS is not None
    for provider in model_configs.GENERATE_TEXT_MODELS:
      for model in model_configs.GENERATE_TEXT_MODELS[provider]:
        assert isinstance(
            model_configs.GENERATE_TEXT_MODELS[provider][model],
            types.ProviderModelType)
        assert model_configs.GENERATE_TEXT_MODELS[
            provider][model].provider == provider
        assert model_configs.GENERATE_TEXT_MODELS[
            provider][model].model == model

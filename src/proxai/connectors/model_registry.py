import functools
from typing import Callable
import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai as openai_provider
import proxai.connectors.providers.claude as claude_provider
import proxai.connectors.providers.gemini as gemini_provider
import proxai.connectors.providers.cohere_api as cohere_api_provider
import proxai.connectors.providers.databricks as databricks_provider
import proxai.connectors.providers.mistral as mistral_provider
import proxai.connectors.providers.huggingface as huggingface_provider
import proxai.connectors.providers.mock_provider as mock_provider
import proxai.connectors.providers.deepseek as deepseek_provider
import proxai.connectors.providers.grok as grok_provider
import proxai.types as types
import proxai.connectors.model_configs as model_configs

_MODEL_CONNECTOR_MAP = {
  'openai': openai_provider.OpenAIConnector,
  'claude': claude_provider.ClaudeConnector,
  'gemini': gemini_provider.GeminiConnector,
  'cohere': cohere_api_provider.CohereConnector,
  'databricks': databricks_provider.DatabricksConnector,
  'mistral': mistral_provider.MistralConnector,
  'huggingface': huggingface_provider.HuggingFaceConnector,
  'deepseek': deepseek_provider.DeepSeekConnector,
  'grok': grok_provider.GrokConnector,
  'mock_provider': mock_provider.MockProviderModelConnector,
  'mock_failing_provider': mock_provider.MockFailingProviderModelConnector,
  'mock_slow_provider': mock_provider.MockSlowProviderModelConnector,
}


def get_model_connector(
    provider_model_config: types.ProviderModelConfigsType,
    without_additional_args: bool = False
) -> Callable[[], model_connector.ProviderModelConnector]:
  provider_model = provider_model_config.provider_model
  if provider_model.provider not in _MODEL_CONNECTOR_MAP:
    raise ValueError(f'Provider not supported. {provider_model.provider}')
  connector = _MODEL_CONNECTOR_MAP[provider_model.provider]
  if without_additional_args:
    return connector
  return functools.partial(
      connector,
      init_from_params=model_connector.ProviderModelConnectorParams(
          provider_model=provider_model,
          provider_model_config=provider_model_config))

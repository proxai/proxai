import functools
from typing import Callable
import proxai.connectors.model_connector as model_connector
import proxai.connectors.openai as openai_provider
import proxai.connectors.claude as claude_provider
import proxai.connectors.gemini as gemini_provider
import proxai.connectors.cohere_api as cohere_api_provider
import proxai.connectors.databricks as databricks_provider
import proxai.connectors.mistral as mistral_provider
import proxai.connectors.hugging_face as hugging_face_provider
import proxai.connectors.mock_provider as mock_provider
import proxai.types as types
import proxai.connectors.model_configs as model_configs

_MODEL_CONNECTOR_MAP = {
  'openai': openai_provider.OpenAIConnector,
  'claude': claude_provider.ClaudeConnector,
  'gemini': gemini_provider.GeminiConnector,
  'cohere': cohere_api_provider.CohereConnector,
  'databricks': databricks_provider.DatabricksConnector,
  'mistral': mistral_provider.MistralConnector,
  'hugging_face': hugging_face_provider.HuggingFaceConnector,
  'mock_provider': mock_provider.MockProviderModelConnector,
  'mock_failing_provider': mock_provider.MockFailingProviderModelConnector,
}


def get_model_connector(
    provider_model_identifier: types.ProviderModelIdentifierType
) -> Callable[[], model_connector.ProviderModelConnector]:
  provider_model = model_configs.get_provider_model_config(
      provider_model_identifier)
  if provider_model.provider not in _MODEL_CONNECTOR_MAP:
    raise ValueError(f'Provider not supported. {provider_model.provider}')
  connector = _MODEL_CONNECTOR_MAP[provider_model.provider]
  return functools.partial(connector, provider_model=provider_model)

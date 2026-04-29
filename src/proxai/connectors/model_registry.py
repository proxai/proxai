from __future__ import annotations

import functools
from collections.abc import Callable

import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.claude as claude_provider
import proxai.connectors.providers.cohere as cohere_provider
import proxai.connectors.providers.databricks as databricks_provider
import proxai.connectors.providers.deepseek as deepseek_provider
import proxai.connectors.providers.gemini as gemini_provider
import proxai.connectors.providers.grok as grok_provider
import proxai.connectors.providers.huggingface as huggingface_provider
import proxai.connectors.providers.mistral as mistral_provider
import proxai.connectors.providers.mock_provider as mock_provider
import proxai.connectors.providers.openai as openai_provider

_MODEL_CONNECTOR_MAP = {
    'openai': openai_provider.OpenAIConnector,
    'claude': claude_provider.ClaudeConnector,
    'gemini': gemini_provider.GeminiConnector,
    'cohere': cohere_provider.CohereConnector,
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
    provider: str,
    without_additional_args: bool = False
) -> Callable[[], provider_connector.ProviderConnector]:
  """Return a connector factory for the given provider."""
  if provider not in _MODEL_CONNECTOR_MAP:
    raise ValueError(f'Provider not supported. {provider}')
  connector_cls = _MODEL_CONNECTOR_MAP[provider]
  if without_additional_args:
    return connector_cls
  return functools.partial(
      connector_cls,
      init_from_params=provider_connector.ProviderConnectorParams()
  )

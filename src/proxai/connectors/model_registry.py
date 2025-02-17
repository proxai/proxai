import functools
from typing import Callable
from proxai.connectors.model_connector import ModelConnector
from proxai.connectors.openai import OpenAIConnector
from proxai.connectors.claude import ClaudeConnector
from proxai.connectors.gemini import GeminiConnector
from proxai.connectors.cohere_api import CohereConnector
from proxai.connectors.databricks import DatabricksConnector
from proxai.connectors.mistral import MistralConnector
from proxai.connectors.hugging_face import HuggingFaceConnector
import proxai.connectors.mock_model_connector as mock_model_connector
import proxai.types as types

_MODEL_CONNECTOR_MAP = {
  types.Provider.OPENAI: OpenAIConnector,
  types.Provider.CLAUDE: ClaudeConnector,
  types.Provider.GEMINI: GeminiConnector,
  types.Provider.COHERE: CohereConnector,
  types.Provider.DATABRICKS: DatabricksConnector,
  types.Provider.MISTRAL: MistralConnector,
  types.Provider.HUGGING_FACE: HuggingFaceConnector,
  types.Provider.MOCK_PROVIDER: mock_model_connector.MockModelConnector,
  types.Provider.MOCK_FAILING_PROVIDER: (
      mock_model_connector.MockFailingConnector),
}


def get_model_connector(model: types.ModelType) -> Callable[[], ModelConnector]:
  provider, _ = model
  if provider not in _MODEL_CONNECTOR_MAP:
    raise ValueError(f'Provider not supported. {provider}')
  connector = _MODEL_CONNECTOR_MAP[provider]
  return functools.partial(connector, model=model)

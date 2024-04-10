import dataclasses
import os
from typing import Dict, Optional
import proxai.types as types
import proxai.type_utils as type_utils
from proxai.connectors.model_connector import ModelConnector
from proxai.connectors.openai import OpenAIConnector
from proxai.connectors.claude import ClaudeConnector
from proxai.connectors.gemini import GeminiConnector
from proxai.connectors.cohere_api import CohereConnector
from proxai.connectors.databricks import DatabricksConnector
from proxai.connectors.mistral import MistralConnector
from proxai.connectors.hugging_face import HuggingFaceConnector
from proxai.logging.utils import LocalLoggingOptions

_RUN_TYPE: types.RunType = types.RunType.PRODUCTION
_REGISTERED_VALUES: Dict[str, types.ValueType] = {}
_INITIALIZED_MODEL_CONNECTORS: Dict[types.ModelType, ModelConnector] = {}
_LOCAL_LOGGING_OPTIONS: LocalLoggingOptions = LocalLoggingOptions()


def _set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type


@dataclasses.dataclass
class LoggingOptions:
  time: bool = True
  prompt: bool = True
  response: bool = True
  error: bool = True


def connect(
    logging_path: str=None,
    logging_options: LoggingOptions=None):
  global _LOCAL_LOGGING_OPTIONS
  if logging_path:
    _LOCAL_LOGGING_OPTIONS.path = logging_path
  if logging_options:
    _LOCAL_LOGGING_OPTIONS.time = logging_options.time
    _LOCAL_LOGGING_OPTIONS.prompt = logging_options.prompt
    _LOCAL_LOGGING_OPTIONS.response = logging_options.response
    _LOCAL_LOGGING_OPTIONS.error = logging_options.error


def _init_model_connector(model: types.ModelType) -> ModelConnector:
  global _LOCAL_LOGGING_OPTIONS
  provider, _ = model
  connector = None
  if provider == types.Provider.OPENAI:
    connector =  OpenAIConnector
  elif provider == types.Provider.CLAUDE:
    connector =  ClaudeConnector
  elif provider == types.Provider.GEMINI:
    connector =  GeminiConnector
  elif provider == types.Provider.COHERE:
    connector =  CohereConnector
  elif provider == types.Provider.DATABRICKS:
    connector =  DatabricksConnector
  elif provider == types.Provider.MISTRAL:
    connector =  MistralConnector
  elif provider == types.Provider.HUGGING_FACE:
    connector =  HuggingFaceConnector
  else:
    raise ValueError(f'Provider not supported. {model}')

  if _LOCAL_LOGGING_OPTIONS.path:
    return connector(
        model=model,
        run_type=_RUN_TYPE,
        logging_options=_LOCAL_LOGGING_OPTIONS)

  return connector(
      model=model,
      run_type=_RUN_TYPE)


def _get_model_connector(value_name: str) -> ModelConnector:
  global _REGISTERED_VALUES
  global _INITIALIZED_MODEL_CONNECTORS
  if value_name == types.GENERATE_TEXT:
    if types.GENERATE_TEXT not in _REGISTERED_VALUES:
      default_model = (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)
      _REGISTERED_VALUES[types.GENERATE_TEXT] = default_model
    if (_REGISTERED_VALUES[types.GENERATE_TEXT]
        not in _INITIALIZED_MODEL_CONNECTORS):
      _INITIALIZED_MODEL_CONNECTORS[_REGISTERED_VALUES[types.GENERATE_TEXT]] = (
          _init_model_connector(_REGISTERED_VALUES[types.GENERATE_TEXT]))
    return _INITIALIZED_MODEL_CONNECTORS[
        _REGISTERED_VALUES[types.GENERATE_TEXT]]
  raise ValueError(
      f'Value name not supported: {value_name}.\n'
      'Supported value names: {types.GENERATE_TEXT}')


def set_model(generate_text: types.ModelType=None):
  global _REGISTERED_VALUES
  if generate_text:
    type_utils.check_model_type(generate_text)
    _REGISTERED_VALUES[types.GENERATE_TEXT] = generate_text


def generate_text(
    prompt: str,
    max_tokens: int = 100) -> str:
  model_connector = _get_model_connector(types.GENERATE_TEXT)
  return model_connector.generate_text(prompt, max_tokens)

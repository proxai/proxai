import dataclasses
import os
import random
from typing import Dict, List, Optional, Set
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
import multiprocessing

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


class AvailableModels:
  _generate_text: List[types.ModelType] = []
  _providers_with_key: List[types.Provider] = []
  _working_models: Set[types.ModelType] = set()
  _failed_models: Set[types.ModelType] = set()

  def __init__(self):
    for provider, provider_key_name in types._PROVIDER_KEY_MAP.items():
      provider_flag = True
      for key_name in provider_key_name:
        if key_name not in os.environ:
          provider_flag = False
          break
      if provider_flag:
        self._providers_with_key.append(provider)

  @staticmethod
  def _test_generate_text(
      model: types.ModelType,
      model_connector: ModelConnector):
    try:
      text = model_connector.generate_text(
          prompt=f'What can you say about number {random.randint(1, 1000000)} '
          f'and number {random.randint(1, 1000000)}?',
          max_tokens=100)
      return model, True, text
    except Exception as e:
      return model, False, str(e)

  @property
  def generate_text(self):
    test_models = []
    for provider, models in types.GENERATE_TEXT_MODELS.items():
      for provider_model in models:
        model = (provider, provider_model)
        if model not in _INITIALIZED_MODEL_CONNECTORS:
          _INITIALIZED_MODEL_CONNECTORS[model] = _init_model_connector(model)
        if model not in self._working_models:
          test_models.append(model)

    pool = multiprocessing.Pool(processes=len(test_models))
    results = []
    for test_model in test_models:
      result = pool.apply_async(
          self._test_generate_text,
          args=(test_model, _INITIALIZED_MODEL_CONNECTORS[test_model]))
      results.append(result)
    pool.close()
    pool.join()
    results = [result.get() for result in results]
    for model, status, _ in results:
      if status:
        self._working_models.add(model)
      else:
        self._failed_models.add(model)
    self._generate_text = sorted(list(self._working_models))
    return self._generate_text

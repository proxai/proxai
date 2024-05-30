import copy
import functools
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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
import proxai.caching.query_cache as query_cache
import proxai.serializers.type_serializer as type_serializer
import proxai.stat_types as stat_types
import proxai.connections.available_models as available_models

_RUN_TYPE: types.RunType = types.RunType.PRODUCTION
_REGISTERED_VALUES: Dict[str, types.ModelType] = {}
_INITIALIZED_MODEL_CONNECTORS: Dict[types.ModelType, ModelConnector] = {}
_LOGGING_OPTIONS: types.LoggingOptions = types.LoggingOptions()
_CACHE_OPTIONS: types.CacheOptions = types.CacheOptions()
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager] = None
_STRICT_FEATURE_TEST: bool = False
_STATS: Dict[str, stat_types.RunStats] = {
    stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
    stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
}

CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions


def _init_globals():
  global _REGISTERED_VALUES
  global _INITIALIZED_MODEL_CONNECTORS
  global _LOGGING_OPTIONS
  global _CACHE_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _STRICT_FEATURE_TEST
  global _STATS
  _REGISTERED_VALUES = {}
  _INITIALIZED_MODEL_CONNECTORS = {}
  _LOGGING_OPTIONS = types.LoggingOptions()
  _CACHE_OPTIONS = types.CacheOptions()
  _QUERY_CACHE_MANAGER = None
  _STRICT_FEATURE_TEST = False
  _STATS[stat_types.GlobalStatType.SINCE_CONNECT] = stat_types.RunStats()


def _set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type
  _init_globals()


def connect(
    cache_path: str=None,
    cache_options: CacheOptions=None,
    logging_path: str=None,
    logging_options: LoggingOptions=None,
    strict_feature_test: bool=False):
  global _CACHE_OPTIONS
  global _LOGGING_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _STRICT_FEATURE_TEST
  _init_globals()

  if cache_path and cache_options and cache_options.path:
    raise ValueError('cache_path and cache_options.path are both set.')

  if logging_path and logging_options and logging_options.path:
    raise ValueError('logging_path and logging_options.path are both set.')

  if cache_path:
    _CACHE_OPTIONS.path = cache_path
  if cache_options:
    if cache_options.path:
      _CACHE_OPTIONS.path = cache_options.path
    if cache_options.duration:
      raise ValueError(
          'cache_options.duration is not supported yet.\n'
          'We are looking for contributors! https://github.com/proxai/proxai')
    if cache_options.unique_response_limit:
      _CACHE_OPTIONS.unique_response_limit = cache_options.unique_response_limit
    if cache_options.retry_if_error_cached:
      _CACHE_OPTIONS.retry_if_error_cached = cache_options.retry_if_error_cached
  if _CACHE_OPTIONS.path:
    _QUERY_CACHE_MANAGER = query_cache.QueryCacheManager(
        cache_options=_CACHE_OPTIONS)

  if logging_path:
    _LOGGING_OPTIONS.path = logging_path
  if logging_options:
    _LOGGING_OPTIONS.time = logging_options.time
    _LOGGING_OPTIONS.prompt = logging_options.prompt
    _LOGGING_OPTIONS.response = logging_options.response
    _LOGGING_OPTIONS.error = logging_options.error

  _STRICT_FEATURE_TEST = strict_feature_test


def _init_model_connector(model: types.ModelType) -> ModelConnector:
  global _LOGGING_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _STATS
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

  connector = functools.partial(
      connector,
      model=model,
      run_type=_RUN_TYPE,
      strict_feature_test=_STRICT_FEATURE_TEST,
      stats=_STATS)

  if _QUERY_CACHE_MANAGER:
    connector = functools.partial(
        connector,
        query_cache_manager=_QUERY_CACHE_MANAGER)

  if _LOGGING_OPTIONS.path:
    connector = functools.partial(
        connector,
        logging_options=_LOGGING_OPTIONS)

  return connector()


def _get_model_connector(
    call_type: types.CallType,
    model: Optional[types.ModelType]=None) -> ModelConnector:
  global _REGISTERED_VALUES
  global _INITIALIZED_MODEL_CONNECTORS
  if call_type == types.CallType.GENERATE_TEXT:
    if call_type not in _REGISTERED_VALUES:
      default_model = (types.Provider.OPENAI, types.OpenAIModel.GPT_3_5_TURBO)
      _REGISTERED_VALUES[call_type] = default_model
    if model == None:
      model = _REGISTERED_VALUES[call_type]
    if model not in _INITIALIZED_MODEL_CONNECTORS:
      _INITIALIZED_MODEL_CONNECTORS[model] = _init_model_connector(model)
    return _INITIALIZED_MODEL_CONNECTORS[model]


def set_model(generate_text: types.ModelType=None):
  global _REGISTERED_VALUES
  if generate_text:
    type_utils.check_model_type(generate_text)
    _REGISTERED_VALUES[types.CallType.GENERATE_TEXT] = generate_text


def generate_text(
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    messages: Optional[types.MessagesType] = None,
    max_tokens: Optional[int] = 100,
    temperature: Optional[float] = None,
    stop: Optional[types.StopType] = None,
    model: Optional[types.ModelType] = None,
    use_cache: bool = True,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False) -> str:
  if prompt != None and messages != None:
    raise ValueError('prompt and messages cannot be set at the same time.')
  if messages != None:
    type_utils.check_messages_type(messages)
  model_connector = _get_model_connector(
      types.CallType.GENERATE_TEXT,
      model=model)
  logging_record = model_connector.generate_text(
      prompt=prompt,
      system=system,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop,
      model=model,
      use_cache=use_cache,
      unique_response_limit=unique_response_limit)
  if logging_record.response_record.error:
    error_traceback = ''
    if logging_record.response_record.error_traceback:
      error_traceback = logging_record.response_record.error_traceback + '\n'
    raise Exception(error_traceback + logging_record.response_record.error)
  if extensive_return:
    return logging_record
  return logging_record.response_record.response


def get_summary(
    run_time: bool = False,
    json: bool = False) -> Union[stat_types.RunStats, Dict[str, Any]]:
  stat_value = None
  if run_time:
    stat_value = copy.deepcopy(_STATS[stat_types.GlobalStatType.RUN_TIME])
  else:
    stat_value = copy.deepcopy(_STATS[stat_types.GlobalStatType.SINCE_CONNECT])
  if json:
    return type_serializer.encode_run_stats(stat_value)

  class StatValue(stat_types.RunStats):
    def __init__(self, stat_value):
      super().__init__(**stat_value.__dict__)

    def serialize(self):
      return type_serializer.encode_run_stats(self)

  return StatValue(stat_value)


def _get_cache_options() -> CacheOptions:
  return _CACHE_OPTIONS


def _get_initialized_model_connectors()-> Dict[
    types.ModelType, ModelConnector]:
  return _INITIALIZED_MODEL_CONNECTORS


def get_available_models() -> available_models.AvailableModels:
  return available_models.AvailableModels(
      get_cache_options=_get_cache_options,
      get_initialized_model_connectors=_get_initialized_model_connectors,
      init_model_connector=_init_model_connector)

import copy
import functools
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
import proxai.connections.proxdash as proxdash
import proxai.experiment.experiment as experiment
from proxai.logging.utils import log_proxdash_message

_RUN_TYPE: types.RunType = types.RunType.PRODUCTION
_HIDDEN_RUN_KEY: Optional[str] = None
_EXPERIMENT_PATH: Optional[str] = None
_REGISTERED_VALUES: Dict[str, types.ModelType] = {}
_INITIALIZED_MODEL_CONNECTORS: Dict[types.ModelType, ModelConnector] = {}
_LOGGING_OPTIONS: types.LoggingOptions = types.LoggingOptions()
_CACHE_OPTIONS: types.CacheOptions = types.CacheOptions()
_PROXDASH_OPTIONS: types.ProxDashOptions = types.ProxDashOptions()
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager] = None
_STRICT_FEATURE_TEST: bool = False
_STATS: Dict[str, stat_types.RunStats] = {
    stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
    stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
}
_PROXDASH_CONNECTION: Optional[proxdash.ProxDashConnection] = None
_ALLOW_MULTIPROCESSING: bool = False

CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions

def _init_hidden_run_key():
  global _HIDDEN_RUN_KEY
  if _HIDDEN_RUN_KEY == None:
    _HIDDEN_RUN_KEY = experiment.get_hidden_run_key()


def _init_experiment_path(experiment_path: Optional[str]):
  global _EXPERIMENT_PATH
  if not experiment_path:
    return
  experiment.validate_experiment_path(experiment_path)
  _EXPERIMENT_PATH = experiment_path


def _init_globals():
  global _REGISTERED_VALUES
  global _INITIALIZED_MODEL_CONNECTORS
  global _LOGGING_OPTIONS
  global _CACHE_OPTIONS
  global _PROXDASH_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _STRICT_FEATURE_TEST
  global _ALLOW_MULTIPROCESSING
  global _STATS
  _REGISTERED_VALUES = {}
  _INITIALIZED_MODEL_CONNECTORS = {}
  _LOGGING_OPTIONS = types.LoggingOptions()
  _CACHE_OPTIONS = types.CacheOptions()
  _PROXDASH_OPTIONS = types.ProxDashOptions()
  _QUERY_CACHE_MANAGER = None
  _STRICT_FEATURE_TEST = False
  _ALLOW_MULTIPROCESSING = False
  _STATS[stat_types.GlobalStatType.SINCE_CONNECT] = stat_types.RunStats()


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
      run_type=_get_run_type(),
      strict_feature_test=_STRICT_FEATURE_TEST,
      stats=_STATS,
      get_logging_options=_get_logging_options,
      get_proxdash_connection=_get_proxdash_connection)

  if _QUERY_CACHE_MANAGER:
    connector = functools.partial(
        connector,
        query_cache_manager=_QUERY_CACHE_MANAGER)

  return connector()


def _get_cache_options() -> CacheOptions:
  return _CACHE_OPTIONS


def _get_logging_options() -> LoggingOptions:
  return _LOGGING_OPTIONS


def _get_proxdash_options() -> ProxDashOptions:
  return _PROXDASH_OPTIONS


def _get_experiment_path() -> str:
  if _EXPERIMENT_PATH:
    return _EXPERIMENT_PATH
  return '(not set)'


def _get_hidden_run_key() -> str:
  if not _HIDDEN_RUN_KEY:
    _init_hidden_run_key()
  return _HIDDEN_RUN_KEY


def _get_initialized_model_connectors()-> Dict[
    types.ModelType, ModelConnector]:
  return _INITIALIZED_MODEL_CONNECTORS


def _get_proxdash_connection() -> proxdash.ProxDashConnection:
  global _PROXDASH_CONNECTION
  if not _PROXDASH_CONNECTION:
    _PROXDASH_CONNECTION = proxdash.ProxDashConnection(
        hidden_run_key=_get_hidden_run_key(),
        get_logging_options=_get_logging_options,
        get_proxdash_options=_get_proxdash_options)
  _PROXDASH_CONNECTION.experiment_path = _get_experiment_path()
  return _PROXDASH_CONNECTION


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


def _get_allow_multiprocessing() -> bool:
  return _ALLOW_MULTIPROCESSING


def _set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type
  _init_globals()


def _get_run_type() -> types.RunType:
  return _RUN_TYPE


def check_health(
    experiment_path: Optional[str]='check_health',
    verbose: bool = False,
    allow_multiprocessing: bool = False
) -> Tuple[List[types.ModelType], List[types.ModelType]]:
  experiment.validate_experiment_path(experiment_path)
  logging_options = types.LoggingOptions()
  proxdash_options = types.ProxDashOptions(stdout=True)
  proxdash_connection = proxdash.ProxDashConnection(
      hidden_run_key=_get_hidden_run_key(),
      logging_options=logging_options,
      proxdash_options=proxdash_options)
  proxdash_connection.experiment_path = experiment_path
  log_proxdash_message(
      logging_options=logging_options,
      proxdash_options=proxdash_options,
      message='Starting to test each model...',
      type=types.LoggingType.INFO)
  models = available_models.AvailableModels(
      cache_options=types.CacheOptions(),
      logging_options=logging_options,
      proxdash_connection=proxdash_connection,
      allow_multiprocessing=allow_multiprocessing,
      get_initialized_model_connectors=_get_initialized_model_connectors,
      init_model_connector=_init_model_connector)
  succeeded_models, failed_models = models.generate_text(
      verbose=verbose, return_all=True)
  log_proxdash_message(
      logging_options=logging_options,
      proxdash_options=proxdash_options,
      message=f'Finished testing. Succeeded Models: {len(succeeded_models)}, '
              f'Failed Models: {len(failed_models)}',
      type=types.LoggingType.INFO)
  if proxdash_connection.status == types.ProxDashConnectionStatus.CONNECTED:
    log_proxdash_message(
        logging_options=logging_options,
        proxdash_options=proxdash_options,
        message='Results are uploaded to the ProxDash.',
        type=types.LoggingType.INFO)
  return succeeded_models, failed_models


def connect(
    experiment_path: Optional[str]=None,
    cache_path: str=None,
    cache_options: CacheOptions=None,
    logging_path: str=None,
    logging_options: LoggingOptions=None,
    proxdash_options: ProxDashOptions=None,
    allow_multiprocessing: bool=False,
    strict_feature_test: bool=False):
  global _CACHE_OPTIONS
  global _LOGGING_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _PROXDASH_OPTIONS
  global _STRICT_FEATURE_TEST
  global _ALLOW_MULTIPROCESSING
  _init_globals()
  _init_experiment_path(experiment_path)

  if cache_path and cache_options and cache_options.path:
    raise ValueError('cache_path and cache_options.path are both set.')

  if logging_path and logging_options and logging_options.logging_path:
    raise ValueError('logging_path and logging_options.logging_path are '
                     'both set.')

  _ALLOW_MULTIPROCESSING = allow_multiprocessing

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
    _LOGGING_OPTIONS.logging_path = logging_path
  if logging_options:
    _LOGGING_OPTIONS.stdout = logging_options.stdout
    _LOGGING_OPTIONS.hide_sensitive_content = (
        logging_options.hide_sensitive_content)

  if proxdash_options:
    _PROXDASH_OPTIONS.stdout = proxdash_options.stdout
    _PROXDASH_OPTIONS.hide_sensitive_content = (
        proxdash_options.hide_sensitive_content)
    _PROXDASH_OPTIONS.disable_proxdash = proxdash_options.disable_proxdash

  _STRICT_FEATURE_TEST = strict_feature_test

  _get_proxdash_connection()


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
    provider: Optional[types.Provider] = None,
    model: Optional[types.ProviderModel] = None,
    use_cache: bool = True,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False) -> str:
  if prompt != None and messages != None:
    raise ValueError('prompt and messages cannot be set at the same time.')
  if messages != None:
    type_utils.check_messages_type(messages)

  if (provider is None) != (model is None):
    raise ValueError('provider and model need to be set together.')
  modelValue = None
  if provider is not None and model is not None:
    modelValue = (provider, model)

  model_connector = _get_model_connector(
      types.CallType.GENERATE_TEXT,
      model=modelValue)
  logging_record = model_connector.generate_text(
      prompt=prompt,
      system=system,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop,
      model=modelValue,
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


def get_available_models() -> available_models.AvailableModels:
  return available_models.AvailableModels(
      get_run_type=_get_run_type,
      get_allow_multiprocessing=_get_allow_multiprocessing,
      get_cache_options=_get_cache_options,
      get_logging_options=_get_logging_options,
      get_initialized_model_connectors=_get_initialized_model_connectors,
      get_proxdash_connection=_get_proxdash_connection,
      init_model_connector=_init_model_connector)

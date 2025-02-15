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
_ROOT_LOGGING_PATH: Optional[str] = None
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


def _init_hidden_run_key():
  global _HIDDEN_RUN_KEY
  if _HIDDEN_RUN_KEY == None:
    _HIDDEN_RUN_KEY = experiment.get_hidden_run_key()


def _init_experiment_path(
    experiment_path: Optional[str] = None,
    global_init: Optional[bool] = False
) -> Optional[str]:
  global _EXPERIMENT_PATH
  if not experiment_path:
    if global_init:
      _EXPERIMENT_PATH = None
    return None
  experiment.validate_experiment_path(experiment_path)
  if global_init:
    _EXPERIMENT_PATH = experiment_path
  return experiment_path


def _init_logging_options(
    experiment_path: Optional[str] = None,
    logging_path: Optional[str] = None,
    logging_options: Optional[types.LoggingOptions] = None,
    global_init: Optional[bool] = False
) -> Tuple[types.LoggingOptions, Optional[str]]:
  if logging_path and logging_options and logging_options.logging_path:
    raise ValueError('logging_path and logging_options.logging_path are '
                     'both set. Either set logging_path or '
                     'logging_options.logging_path, but not both.')

  root_logging_path = None
  result_logging_options = types.LoggingOptions()

  if logging_path:
    root_logging_path = logging_path
  elif logging_options and logging_options.logging_path:
    root_logging_path = logging_options.logging_path
  else:
    root_logging_path = None

  if root_logging_path:
    if not os.path.exists(root_logging_path):
      raise ValueError(
          f'Root logging path does not exist: {root_logging_path}')

    if experiment_path:
      result_logging_options.logging_path = os.path.join(
          root_logging_path, experiment_path)
    else:
      result_logging_options.logging_path = root_logging_path
    if not os.path.exists(result_logging_options.logging_path):
      os.makedirs(result_logging_options.logging_path, exist_ok=True)
  else:
    result_logging_options.logging_path = None

  if logging_options:
    result_logging_options.stdout = logging_options.stdout
    result_logging_options.hide_sensitive_content = (
        logging_options.hide_sensitive_content)

  if global_init:
    global _ROOT_LOGGING_PATH
    global _LOGGING_OPTIONS
    _ROOT_LOGGING_PATH = root_logging_path
    _LOGGING_OPTIONS = result_logging_options

  return (result_logging_options, root_logging_path)


def _init_cache_options(
    cache_path: Optional[str] = None,
    cache_options: Optional[types.CacheOptions] = None,
    global_init: Optional[bool] = False
) -> Tuple[types.CacheOptions, Optional[query_cache.QueryCacheManager]]:
  if cache_path and cache_options and cache_options.cache_path:
    raise ValueError('cache_path and cache_options.cache_path are both set.'
                     'Either set cache_path or cache_options.cache_path, but '
                     'not both.')

  result_cache_options = types.CacheOptions()
  result_query_cache_manager = None

  if cache_path:
    result_cache_options.cache_path = cache_path
  if cache_options:
    if cache_options.cache_path:
      result_cache_options.cache_path = cache_options.cache_path
    if cache_options.duration:
      raise ValueError(
          'cache_options.duration is not supported yet.\n'
          'We are looking for contributors! https://github.com/proxai/proxai')
    result_cache_options.unique_response_limit = (
        cache_options.unique_response_limit)
    result_cache_options.retry_if_error_cached = (
        cache_options.retry_if_error_cached)
    result_cache_options.clear_query_cache_on_connect = (
        cache_options.clear_query_cache_on_connect)
    result_cache_options.clear_model_cache_on_connect = (
        cache_options.clear_model_cache_on_connect)
  if result_cache_options.cache_path:
    result_query_cache_manager = query_cache.QueryCacheManager(
        cache_options=result_cache_options)
  if global_init:
    global _CACHE_OPTIONS
    global _QUERY_CACHE_MANAGER
    _CACHE_OPTIONS = result_cache_options
    _QUERY_CACHE_MANAGER = result_query_cache_manager
  return (result_cache_options, result_query_cache_manager)


def _init_proxdash_options(
    proxdash_options: Optional[types.ProxDashOptions] = None,
    global_init: Optional[bool] = False) -> types.ProxDashOptions:
  result_proxdash_options = types.ProxDashOptions()
  if proxdash_options:
    result_proxdash_options.stdout = proxdash_options.stdout
    result_proxdash_options.hide_sensitive_content = (
        proxdash_options.hide_sensitive_content)
    result_proxdash_options.disable_proxdash = proxdash_options.disable_proxdash

  if global_init:
    global _PROXDASH_OPTIONS
    _PROXDASH_OPTIONS = result_proxdash_options
  return result_proxdash_options


def _init_allow_multiprocessing(
    allow_multiprocessing: Optional[bool] = None,
    global_init: Optional[bool] = False) -> Optional[bool]:
  if allow_multiprocessing is None:
    return None
  if global_init:
    global _ALLOW_MULTIPROCESSING
    _ALLOW_MULTIPROCESSING = allow_multiprocessing
  return allow_multiprocessing


def _init_strict_feature_test(
    strict_feature_test: Optional[bool] = None,
    global_init: Optional[bool] = False) -> Optional[bool]:
  if strict_feature_test is None:
    return None
  if global_init:
    global _STRICT_FEATURE_TEST
    _STRICT_FEATURE_TEST = strict_feature_test
  return strict_feature_test


def _init_model_connector(model: types.ModelType) -> ModelConnector:
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
        get_experiment_path=_get_experiment_path,
        get_logging_options=_get_logging_options,
        get_proxdash_options=_get_proxdash_options)
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


def _get_run_type() -> types.RunType:
  return _RUN_TYPE


def _set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type
  _init_globals()


def check_health(
    experiment_path: Optional[str]='check_health',
    verbose: bool = False,
    allow_multiprocessing: bool = False
) -> Tuple[List[types.ModelType], List[types.ModelType]]:
  experiment_path = _init_experiment_path(experiment_path=experiment_path)
  logging_options, _ = _init_logging_options(
      experiment_path=experiment_path,
      logging_options=types.LoggingOptions())
  cache_options, _ = _init_cache_options()
  proxdash_options = _init_proxdash_options(
      proxdash_options=types.ProxDashOptions(stdout=True))
  allow_multiprocessing = _init_allow_multiprocessing(
      allow_multiprocessing=allow_multiprocessing)

  proxdash_connection = proxdash.ProxDashConnection(
      hidden_run_key=_get_hidden_run_key(),
      experiment_path=experiment_path,
      proxdash_options=proxdash_options)
  log_proxdash_message(
      logging_options=logging_options,
      proxdash_options=proxdash_options,
      message='Starting to test each model...',
      type=types.LoggingType.INFO)
  models = available_models.AvailableModels(
      proxdash_connection=proxdash_connection,
      allow_multiprocessing=allow_multiprocessing,
      cache_options=cache_options,
      logging_options=logging_options,
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
  _init_globals()
  _init_experiment_path(
      experiment_path=experiment_path,
      global_init=True)
  _init_logging_options(
      experiment_path=experiment_path,
      logging_path=logging_path,
      logging_options=logging_options,
      global_init=True)
  _init_cache_options(
      cache_path=cache_path,
      cache_options=cache_options,
      global_init=True)
  _init_proxdash_options(
      proxdash_options=proxdash_options,
      global_init=True)
  _init_allow_multiprocessing(
      allow_multiprocessing=allow_multiprocessing,
      global_init=True)
  _init_strict_feature_test(
      strict_feature_test=strict_feature_test,
      global_init=True)

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
    extensive_return: bool = False) -> Union[str, types.LoggingRecord]:
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

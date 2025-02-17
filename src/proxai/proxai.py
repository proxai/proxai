import copy
import datetime
import functools
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import proxai.types as types
import proxai.type_utils as type_utils
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_registry as model_registry
import proxai.caching.query_cache as query_cache
import proxai.caching.model_cache as model_cache
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
_INITIALIZED_MODEL_CONNECTORS: Dict[
    types.ModelType, model_connector.ModelConnector] = {}
_ROOT_LOGGING_PATH: Optional[str] = None
_LOGGING_OPTIONS: types.LoggingOptions = types.LoggingOptions()
_CACHE_OPTIONS: types.CacheOptions = types.CacheOptions()
_PROXDASH_OPTIONS: types.ProxDashOptions = types.ProxDashOptions()
_DEFAULT_MODEL_CACHE_PATH: Optional[tempfile.TemporaryDirectory] = None
_DEFAULT_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager] = None
_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager] = None
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager] = None
_STRICT_FEATURE_TEST: bool = False
_SUPPRESS_PROVIDER_ERRORS: bool = False
_STATS: Dict[str, stat_types.RunStats] = {
    stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
    stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
}
_PROXDASH_CONNECTION: Optional[proxdash.ProxDashConnection] = None
_ALLOW_MULTIPROCESSING: bool = True

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
  global _MODEL_CACHE_MANAGER
  global _STRICT_FEATURE_TEST
  global _SUPPRESS_PROVIDER_ERRORS
  global _ALLOW_MULTIPROCESSING
  global _STATS
  global _DEFAULT_MODEL_CACHE_MANAGER
  global _DEFAULT_MODEL_CACHE_PATH
  _REGISTERED_VALUES = {}
  _INITIALIZED_MODEL_CONNECTORS = {}
  _LOGGING_OPTIONS = types.LoggingOptions()
  _CACHE_OPTIONS = types.CacheOptions()
  _PROXDASH_OPTIONS = types.ProxDashOptions()
  _QUERY_CACHE_MANAGER = None
  _MODEL_CACHE_MANAGER = None
  _STRICT_FEATURE_TEST = False
  _SUPPRESS_PROVIDER_ERRORS = False
  _ALLOW_MULTIPROCESSING = True
  _STATS[stat_types.GlobalStatType.SINCE_CONNECT] = stat_types.RunStats()
  _DEFAULT_MODEL_CACHE_MANAGER = None
  _DEFAULT_MODEL_CACHE_PATH = None


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
) -> Tuple[
    types.CacheOptions,
    Optional[model_cache.ModelCacheManager],
    Optional[query_cache.QueryCacheManager]]:
  if cache_path and cache_options and cache_options.cache_path:
    raise ValueError('cache_path and cache_options.cache_path are both set.'
                     'Either set cache_path or cache_options.cache_path, but '
                     'not both.')

  result_cache_options = types.CacheOptions()
  result_model_cache_manager = None
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
    result_model_cache_manager = model_cache.ModelCacheManager(
        cache_options=result_cache_options)
    result_query_cache_manager = query_cache.QueryCacheManager(
        cache_options=result_cache_options)
  if global_init:
    global _CACHE_OPTIONS
    global _MODEL_CACHE_MANAGER
    global _QUERY_CACHE_MANAGER
    _CACHE_OPTIONS = result_cache_options
    _MODEL_CACHE_MANAGER = result_model_cache_manager
    _QUERY_CACHE_MANAGER = result_query_cache_manager
  return (
      result_cache_options,
      result_model_cache_manager,
      result_query_cache_manager)


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


def _init_suppress_errors(
    suppress_provider_errors: Optional[bool] = None,
    global_init: Optional[bool] = False) -> Optional[bool]:
  if suppress_provider_errors is None:
    return None
  if global_init:
    global _SUPPRESS_PROVIDER_ERRORS
    _SUPPRESS_PROVIDER_ERRORS = suppress_provider_errors
  return suppress_provider_errors


def _init_model_connector(
    model: types.ModelType) -> model_connector.ModelConnector:
  global _QUERY_CACHE_MANAGER
  global _STATS
  connector = model_registry.get_model_connector(model)
  connector = functools.partial(
      connector,
      get_query_cache_manager=_get_query_cache_manager)
  return connector(
      run_type=_get_run_type(),
      strict_feature_test=_get_strict_feature_test(),
      stats=_STATS,
      get_logging_options=_get_logging_options,
      get_proxdash_connection=_get_proxdash_connection)


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


def _get_initialized_model_connectors() -> Dict[
    types.ModelType, model_connector.ModelConnector]:
  return _INITIALIZED_MODEL_CONNECTORS


def _get_model_cache_manager() -> model_cache.ModelCacheManager:
  global _MODEL_CACHE_MANAGER
  global _DEFAULT_MODEL_CACHE_MANAGER
  global _DEFAULT_MODEL_CACHE_PATH
  if _MODEL_CACHE_MANAGER is not None:
    return _MODEL_CACHE_MANAGER
  if _DEFAULT_MODEL_CACHE_PATH is None:
    default_cache_path = tempfile.TemporaryDirectory()
    _DEFAULT_MODEL_CACHE_PATH = default_cache_path
    _DEFAULT_MODEL_CACHE_MANAGER = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(
            cache_path=default_cache_path.name))
  return _DEFAULT_MODEL_CACHE_MANAGER


def _get_query_cache_manager() -> query_cache.QueryCacheManager:
  return _QUERY_CACHE_MANAGER


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
    model: Optional[types.ModelType]=None) -> model_connector.ModelConnector:
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


def _get_strict_feature_test() -> bool:
  return _STRICT_FEATURE_TEST


def _get_suppress_provider_errors() -> bool:
  return _SUPPRESS_PROVIDER_ERRORS


def _get_run_type() -> types.RunType:
  return _RUN_TYPE


def set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type
  _init_globals()


def check_health(
    experiment_path: Optional[str]=None,
    verbose: bool = True,
    allow_multiprocessing: bool = True
) -> types.ModelStatus:
  if not experiment_path:
    now = datetime.datetime.now()
    experiment_path = (
        f'connection_health/{now.strftime("%Y-%m-%d_%H-%M-%S")}')
  experiment_path = _init_experiment_path(experiment_path=experiment_path)
  logging_options, _ = _init_logging_options(
      experiment_path=experiment_path,
      logging_options=types.LoggingOptions())
  if _get_run_type() == types.RunType.TEST:
    proxdash_options = types.ProxDashOptions(
        stdout=False,
        disable_proxdash=True)
  else:
    proxdash_options = types.ProxDashOptions(stdout=True)
  allow_multiprocessing = _init_allow_multiprocessing(
      allow_multiprocessing=allow_multiprocessing)

  proxdash_connection = proxdash.ProxDashConnection(
      hidden_run_key=_get_hidden_run_key(),
      experiment_path=experiment_path,
      proxdash_options=proxdash_options)
  if verbose:
    print('> Starting to test each model...')
  models = available_models.AvailableModels(
      get_run_type=_get_run_type,
      proxdash_connection=proxdash_connection,
      allow_multiprocessing=allow_multiprocessing,
      logging_options=logging_options,
      get_initialized_model_connectors=_get_initialized_model_connectors,
      init_model_connector=_init_model_connector)
  model_status = models.generate_text(
      verbose=verbose, return_all=True)
  if verbose:
    providers = set(
        [model[0] for model in model_status.working_models] +
        [model[0] for model in model_status.failed_models])
    model_query_map = {
        query.query_record.model: query
        for query in model_status.provider_queries
    }
    result_table = {
        provider: {'working': [], 'failed': []} for provider in providers}
    for model in model_status.working_models:
      result_table[model[0]]['working'].append(model[1])
    for model in model_status.failed_models:
      result_table[model[0]]['failed'].append(model[1])
    print('> Finished testing.\n'
          f'   Registered Providers: {len(providers)}\n'
          f'   Succeeded Models: {len(model_status.working_models)}\n'
          f'   Failed Models: {len(model_status.failed_models)}')
    for provider in sorted(providers):
      print(f'> {provider}:')
      for model in sorted(result_table[provider]['working']):
        duration = model_query_map[
            (provider, model)].response_record.response_time
        print(f'   [ WORKING | {duration.total_seconds():6.2f}s ]: {model}')
      for model in sorted(result_table[provider]['failed']):
        duration = model_query_map[
            (provider, model)].response_record.response_time
        print(f'   [ FAILED  | {duration.total_seconds():6.2f}s ]: {model}')
  if proxdash_connection.status == types.ProxDashConnectionStatus.CONNECTED:
    log_proxdash_message(
        logging_options=logging_options,
        proxdash_options=proxdash_options,
        message='Results are uploaded to the ProxDash.',
        type=types.LoggingType.INFO)
  return model_status


def connect(
    experiment_path: Optional[str]=None,
    cache_path: str=None,
    cache_options: CacheOptions=None,
    logging_path: str=None,
    logging_options: LoggingOptions=None,
    proxdash_options: ProxDashOptions=None,
    allow_multiprocessing: bool=True,
    strict_feature_test: bool=False,
    suppress_provider_errors: bool=False):
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
  _init_suppress_errors(
      suppress_provider_errors=suppress_provider_errors,
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
    use_cache: Optional[bool] = None,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False,
    suppress_provider_errors: Optional[bool] = None) -> Union[str, types.LoggingRecord]:
  if prompt != None and messages != None:
    raise ValueError('prompt and messages cannot be set at the same time.')
  if messages != None:
    type_utils.check_messages_type(messages)

  if (provider is None) != (model is None):
    raise ValueError('provider and model need to be set together.')
  modelValue = None
  if provider is not None and model is not None:
    modelValue = (provider, model)

  if use_cache and not _get_query_cache_manager():
    raise ValueError(
        'use_cache is True but query cache is not initialized. '
        'Please set cache_path on px.connect() to use query cache.')
  if use_cache is None:
    use_cache = True

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
    if suppress_provider_errors or (
        suppress_provider_errors is None and _get_suppress_provider_errors()):
      if extensive_return:
        return logging_record
      return logging_record.response_record.error
    else:
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
      get_logging_options=_get_logging_options,
      get_model_cache_manager=_get_model_cache_manager,
      get_initialized_model_connectors=_get_initialized_model_connectors,
      get_proxdash_connection=_get_proxdash_connection,
      init_model_connector=_init_model_connector)


def get_current_options(
    json: bool = False) -> Union[types.RunOptions, Dict[str, Any]]:
  run_options = types.RunOptions(
    run_type=_get_run_type(),
    logging_options=_get_logging_options(),
    cache_options=_get_cache_options(),
    proxdash_options=_get_proxdash_options(),
    allow_multiprocessing=_get_allow_multiprocessing(),
    strict_feature_test=_get_strict_feature_test()
  )
  if json:
    return type_serializer.encode_run_options(run_options=run_options)
  return run_options

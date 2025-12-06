import copy
import datetime
import functools
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
import platformdirs
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
import proxai.connectors.model_configs as model_configs
import proxai.logging.utils as logging_utils

_RUN_TYPE: types.RunType
_HIDDEN_RUN_KEY: str
_EXPERIMENT_PATH: Optional[str]
_ROOT_LOGGING_PATH: Optional[str]
_DEFAULT_MODEL_CACHE_PATH: Optional[tempfile.TemporaryDirectory]
_PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE: bool

_LOGGING_OPTIONS: types.LoggingOptions
_CACHE_OPTIONS: types.CacheOptions
_PROXDASH_OPTIONS: types.ProxDashOptions

_MODEL_CONFIGS: Optional[model_configs.ModelConfigs]
_MODEL_CONFIGS_REQUESTED_FROM_PROXDASH: bool

_REGISTERED_MODEL_CONNECTORS: Dict[
    types.CallType, model_connector.ProviderModelConnector]
_MODEL_CONNECTORS: Dict[
    types.ProviderModelType, model_connector.ProviderModelConnector]
_DEFAULT_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager]
_MODEL_CACHE_MANAGER: Optional[model_cache.ModelCacheManager]
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager]
_PROXDASH_CONNECTION: Optional[proxdash.ProxDashConnection]

_STRICT_FEATURE_TEST: bool
_SUPPRESS_PROVIDER_ERRORS: bool
_ALLOW_MULTIPROCESSING: bool
_MODEL_TEST_TIMEOUT: Optional[int]

_STATS: Dict[stat_types.GlobalStatType, stat_types.RunStats]
_AVAILABLE_MODELS: Optional[available_models.AvailableModels]

CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions
ResponseFormat = types.ResponseFormat


def _init_default_model_cache_manager():
  global _DEFAULT_MODEL_CACHE_PATH
  global _PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE
  global _DEFAULT_MODEL_CACHE_MANAGER
  try:
    app_dirs = platformdirs.PlatformDirs(appname="proxai", appauthor="proxai")
    _DEFAULT_MODEL_CACHE_PATH =  app_dirs.user_cache_dir
    os.makedirs(_DEFAULT_MODEL_CACHE_PATH, exist_ok=True)
    # 4 hours cache duration makes sense for local development if proxai is
    # using platform app cache directory
    _DEFAULT_MODEL_CACHE_MANAGER = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(
            cache_path=_DEFAULT_MODEL_CACHE_PATH,
            model_cache_duration=60 * 60 * 4))
    _PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE = True
  except Exception as e:
    _DEFAULT_MODEL_CACHE_PATH = tempfile.TemporaryDirectory()
    _DEFAULT_MODEL_CACHE_MANAGER = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(
            cache_path=_DEFAULT_MODEL_CACHE_PATH.name))
    _PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE = False


def _init_globals():
  global _RUN_TYPE
  global _HIDDEN_RUN_KEY
  global _EXPERIMENT_PATH
  global _ROOT_LOGGING_PATH

  global _LOGGING_OPTIONS
  global _CACHE_OPTIONS
  global _PROXDASH_OPTIONS

  global _MODEL_CONFIGS
  global _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH

  global _REGISTERED_MODEL_CONNECTORS
  global _MODEL_CONNECTORS
  global _MODEL_CACHE_MANAGER
  global _QUERY_CACHE_MANAGER
  global _PROXDASH_CONNECTION

  global _STRICT_FEATURE_TEST
  global _SUPPRESS_PROVIDER_ERRORS
  global _ALLOW_MULTIPROCESSING
  global _MODEL_TEST_TIMEOUT

  global _STATS
  global _AVAILABLE_MODELS

  _RUN_TYPE = types.RunType.PRODUCTION
  _HIDDEN_RUN_KEY = experiment.get_hidden_run_key()
  _EXPERIMENT_PATH = None
  _ROOT_LOGGING_PATH = None

  _LOGGING_OPTIONS = types.LoggingOptions()
  _CACHE_OPTIONS = types.CacheOptions()
  _PROXDASH_OPTIONS = types.ProxDashOptions()

  _MODEL_CONFIGS = model_configs.ModelConfigs()
  _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH = False

  _REGISTERED_MODEL_CONNECTORS = {}
  _MODEL_CONNECTORS = {}
  _MODEL_CACHE_MANAGER = None
  _QUERY_CACHE_MANAGER = None
  _PROXDASH_CONNECTION = None
  _init_default_model_cache_manager()

  _STRICT_FEATURE_TEST = False
  _SUPPRESS_PROVIDER_ERRORS = False
  _ALLOW_MULTIPROCESSING = True
  _MODEL_TEST_TIMEOUT = 25

  _STATS = {
      stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
      stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
  }
  _AVAILABLE_MODELS = None


def _set_experiment_path(
    experiment_path: Optional[str] = None,
    global_set: Optional[bool] = False
) -> Optional[str]:
  global _EXPERIMENT_PATH
  if experiment_path is None:
    if global_set:
      _EXPERIMENT_PATH = None
    return None
  experiment.validate_experiment_path(experiment_path)
  if global_set:
    _EXPERIMENT_PATH = experiment_path
  return experiment_path


def _set_logging_options(
    experiment_path: Optional[str] = None,
    logging_path: Optional[str] = None,
    logging_options: Optional[types.LoggingOptions] = None,
    global_set: Optional[bool] = False
) -> Tuple[types.LoggingOptions, Optional[str]]:
  if (
      logging_path is not None and
      logging_options is not None and
      logging_options.logging_path is not None):
    raise ValueError('logging_path and logging_options.logging_path are '
                     'both set. Either set logging_path or '
                     'logging_options.logging_path, but not both.')

  root_logging_path = None
  if logging_path:
    root_logging_path = logging_path
  elif logging_options and logging_options.logging_path:
    root_logging_path = logging_options.logging_path
  else:
    root_logging_path = None

  result_logging_options = types.LoggingOptions()
  if root_logging_path is not None:
    if not os.path.exists(root_logging_path):
      raise ValueError(
          f'Root logging path does not exist: {root_logging_path}')

    if experiment_path is not None:
      result_logging_options.logging_path = os.path.join(
          root_logging_path, experiment_path)
    else:
      result_logging_options.logging_path = root_logging_path
    if not os.path.exists(result_logging_options.logging_path):
      os.makedirs(result_logging_options.logging_path, exist_ok=True)
  else:
    result_logging_options.logging_path = None

  if logging_options is not None:
    result_logging_options.stdout = logging_options.stdout
    result_logging_options.hide_sensitive_content = (
        logging_options.hide_sensitive_content)

  if global_set:
    global _ROOT_LOGGING_PATH
    global _LOGGING_OPTIONS
    _ROOT_LOGGING_PATH = root_logging_path
    _LOGGING_OPTIONS = result_logging_options

  return (result_logging_options, root_logging_path)


def _set_cache_options(
    cache_path: Optional[str] = None,
    cache_options: Optional[types.CacheOptions] = None,
    global_set: Optional[bool] = False
) -> types.CacheOptions:
  if (
      cache_path is not None and
      cache_options is not None and
      cache_options.cache_path is not None):
    raise ValueError('cache_path and cache_options.cache_path are both set.'
                     'Either set cache_path or cache_options.cache_path, but '
                     'not both.')

  result_cache_options = types.CacheOptions()
  if cache_path:
    result_cache_options.cache_path = cache_path

  if cache_options:
    if cache_options.cache_path:
      result_cache_options.cache_path = cache_options.cache_path

    result_cache_options.unique_response_limit = (
        cache_options.unique_response_limit)
    result_cache_options.retry_if_error_cached = (
        cache_options.retry_if_error_cached)
    result_cache_options.clear_query_cache_on_connect = (
        cache_options.clear_query_cache_on_connect)

    result_cache_options.disable_model_cache = (
        cache_options.disable_model_cache)
    result_cache_options.clear_model_cache_on_connect = (
        cache_options.clear_model_cache_on_connect)
    result_cache_options.model_cache_duration = (
        cache_options.model_cache_duration)

  if global_set:
    global _CACHE_OPTIONS
    _CACHE_OPTIONS = result_cache_options
  return result_cache_options


def _set_proxdash_options(
    proxdash_options: Optional[types.ProxDashOptions] = None,
    global_set: Optional[bool] = False) -> types.ProxDashOptions:
  result_proxdash_options = types.ProxDashOptions()
  if proxdash_options is not None:
    result_proxdash_options.stdout = proxdash_options.stdout
    result_proxdash_options.hide_sensitive_content = (
        proxdash_options.hide_sensitive_content)
    result_proxdash_options.disable_proxdash = proxdash_options.disable_proxdash
    result_proxdash_options.api_key = proxdash_options.api_key
    result_proxdash_options.base_url = proxdash_options.base_url

  if global_set:
    global _PROXDASH_OPTIONS
    _PROXDASH_OPTIONS = result_proxdash_options
  return result_proxdash_options


def _set_model_configs_requested_from_proxdash(
    model_configs_requested_from_proxdash: Optional[bool] = None,
    global_set: Optional[bool] = False) -> Optional[bool]:
  if model_configs_requested_from_proxdash is None:
    return None
  if global_set:
    global _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH
    _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH = model_configs_requested_from_proxdash
  return model_configs_requested_from_proxdash


def _set_allow_multiprocessing(
    allow_multiprocessing: Optional[bool] = None,
    global_set: Optional[bool] = False) -> Optional[bool]:
  if allow_multiprocessing is None:
    return None
  if global_set:
    global _ALLOW_MULTIPROCESSING
    _ALLOW_MULTIPROCESSING = allow_multiprocessing
  return allow_multiprocessing


def _set_model_test_timeout(
    model_test_timeout: Optional[int] = None,
    global_set: Optional[bool] = False) -> Optional[int]:
  if model_test_timeout is None:
    return None
  if model_test_timeout < 1:
    raise ValueError('model_test_timeout must be greater than 0.')
  if global_set:
    global _MODEL_TEST_TIMEOUT
    _MODEL_TEST_TIMEOUT = model_test_timeout
  return model_test_timeout


def _set_strict_feature_test(
    strict_feature_test: Optional[bool] = None,
    global_set: Optional[bool] = False) -> Optional[bool]:
  if strict_feature_test is None:
    return None
  if global_set:
    global _STRICT_FEATURE_TEST
    _STRICT_FEATURE_TEST = strict_feature_test
  return strict_feature_test


def _set_suppress_provider_errors(
    suppress_provider_errors: Optional[bool] = None,
    global_set: Optional[bool] = False) -> Optional[bool]:
  if suppress_provider_errors is None:
    return None
  if global_set:
    global _SUPPRESS_PROVIDER_ERRORS
    _SUPPRESS_PROVIDER_ERRORS = suppress_provider_errors
  return suppress_provider_errors


def _get_run_type() -> types.RunType:
  return _RUN_TYPE


def _get_model_configs() -> model_configs.ModelConfigs:
  global _MODEL_CONFIGS
  global _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH

  if not _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH:
    model_configs_schema = _get_proxdash_connection().get_model_configs_schema()
    if model_configs_schema is not None:
      _MODEL_CONFIGS.model_configs_schema = model_configs_schema
    _MODEL_CONFIGS_REQUESTED_FROM_PROXDASH = True
  return _MODEL_CONFIGS

def _get_hidden_run_key() -> str:
  return _HIDDEN_RUN_KEY


def _get_experiment_path() -> str:
  return _EXPERIMENT_PATH


def _get_logging_options() -> LoggingOptions:
  return _LOGGING_OPTIONS


def _get_cache_options() -> CacheOptions:
  return _CACHE_OPTIONS


def _get_proxdash_options() -> ProxDashOptions:
  return _PROXDASH_OPTIONS


def _get_model_connector(
    provider_model_identifier: types.ProviderModelIdentifierType
) -> model_connector.ProviderModelConnector:
  global _MODEL_CONNECTORS
  model_configs_instance = _get_model_configs()
  provider_model = model_configs_instance.get_provider_model(
      provider_model_identifier)
  if provider_model in _MODEL_CONNECTORS:
    return _MODEL_CONNECTORS[provider_model]

  connector = model_registry.get_model_connector(
      provider_model,
      model_configs=_get_model_configs())
  _MODEL_CONNECTORS[provider_model] = connector(
      get_run_type=_get_run_type,
      get_strict_feature_test=_get_strict_feature_test,
      get_query_cache_manager=_get_query_cache_manager,
      get_logging_options=_get_logging_options,
      get_proxdash_connection=_get_proxdash_connection,
      stats=_get_stats())
  return _MODEL_CONNECTORS[provider_model]


def _get_registered_model_connector(
    call_type: types.CallType = types.CallType.GENERATE_TEXT
) -> model_connector.ProviderModelConnector:
  global _REGISTERED_MODEL_CONNECTORS
  if call_type not in _REGISTERED_MODEL_CONNECTORS:
    if call_type == types.CallType.GENERATE_TEXT:
      available_models = get_available_models()
      if (
          not available_models.model_cache_manager or
          not available_models.model_cache_manager.get(
              types.CallType.GENERATE_TEXT).working_models):
        print('Checking available models, this may take a while...')
      models = get_available_models().list_models(return_all=True)
      model_configs_instance = _get_model_configs()
      for provider_model in model_configs_instance.get_default_model_priority_list():
        if provider_model in models.working_models:
          _REGISTERED_MODEL_CONNECTORS[call_type] = _get_model_connector(
              provider_model)
          break
      if call_type not in _REGISTERED_MODEL_CONNECTORS:
        if models.working_models:
          _REGISTERED_MODEL_CONNECTORS[call_type] = _get_model_connector(
              models.working_models.pop())
        else:
          raise ValueError(
              'No working models found in current environment:\n'
              '* Please check your environment variables and try again.\n'
              '* You can use px.check_health() method as instructed in '
              'https://www.proxai.co/proxai-docs/check-health')
  return _REGISTERED_MODEL_CONNECTORS[call_type]


def _get_model_cache_manager() -> model_cache.ModelCacheManager:
  global _MODEL_CACHE_MANAGER
  if _MODEL_CACHE_MANAGER is None:
    _MODEL_CACHE_MANAGER = model_cache.ModelCacheManager(
        get_cache_options=_get_cache_options)
  if (_MODEL_CACHE_MANAGER.status !=
      types.ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND):
    return _MODEL_CACHE_MANAGER

  if _DEFAULT_MODEL_CACHE_PATH is not None:
    return _DEFAULT_MODEL_CACHE_MANAGER

  raise ValueError('Model cache manager is not initialized and there is no '
                   'default model cache manager.')


def _get_query_cache_manager() -> query_cache.QueryCacheManager:
  global _QUERY_CACHE_MANAGER
  if _QUERY_CACHE_MANAGER is None:
    _QUERY_CACHE_MANAGER = query_cache.QueryCacheManager(
        get_cache_options=_get_cache_options)

  return _QUERY_CACHE_MANAGER


def _get_proxdash_connection() -> proxdash.ProxDashConnection:
  global _PROXDASH_CONNECTION
  if _PROXDASH_CONNECTION is None:
    _PROXDASH_CONNECTION = proxdash.ProxDashConnection(
        hidden_run_key=_get_hidden_run_key(),
        get_experiment_path=_get_experiment_path,
        get_logging_options=_get_logging_options,
        get_proxdash_options=_get_proxdash_options)
  return _PROXDASH_CONNECTION


def _get_allow_multiprocessing() -> bool:
  return _ALLOW_MULTIPROCESSING


def _get_model_test_timeout() -> int:
  return _MODEL_TEST_TIMEOUT


def _get_strict_feature_test() -> bool:
  return _STRICT_FEATURE_TEST


def _get_suppress_provider_errors() -> bool:
  return _SUPPRESS_PROVIDER_ERRORS


def _get_stats() -> Dict[stat_types.GlobalStatType, stat_types.RunStats]:
  return _STATS


def set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type


def set_model(
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    generate_text: Optional[types.ProviderModelIdentifierType] = None):
  global _REGISTERED_MODEL_CONNECTORS
  if provider_model and generate_text:
    raise ValueError('provider_model and generate_text cannot be set at the '
                     'same time. Please set one of them.')

  if provider_model is None and generate_text is None:
    raise ValueError('provider_model or generate_text must be set.')

  if generate_text:
    provider_model = generate_text

  model_configs_instance = _get_model_configs()
  model_configs_instance.check_provider_model_identifier_type(provider_model)
  _REGISTERED_MODEL_CONNECTORS[
      types.CallType.GENERATE_TEXT] = _get_model_connector(provider_model)


def connect(
    experiment_path: Optional[str]=None,
    cache_path: Optional[str]=None,
    cache_options: Optional[CacheOptions]=None,
    logging_path: Optional[str]=None,
    logging_options: Optional[LoggingOptions]=None,
    proxdash_options: Optional[ProxDashOptions]=None,
    allow_multiprocessing: Optional[bool]=True,
    model_test_timeout: Optional[int]=25,
    strict_feature_test: Optional[bool]=False,
    suppress_provider_errors: Optional[bool]=False):
  _set_experiment_path(
      experiment_path=experiment_path,
      global_set=True)
  _set_logging_options(
      experiment_path=experiment_path,
      logging_path=logging_path,
      logging_options=logging_options,
      global_set=True)
  _set_cache_options(
      cache_path=cache_path,
      cache_options=cache_options,
      global_set=True)
  _set_proxdash_options(
      proxdash_options=proxdash_options,
      global_set=True)
  _set_allow_multiprocessing(
      allow_multiprocessing=allow_multiprocessing,
      global_set=True)
  _set_model_test_timeout(
      model_test_timeout=model_test_timeout,
      global_set=True)
  _set_model_configs_requested_from_proxdash(
      model_configs_requested_from_proxdash=False,
      global_set=True)
  _set_strict_feature_test(
      strict_feature_test=strict_feature_test,
      global_set=True)
  _set_suppress_provider_errors(
      suppress_provider_errors=suppress_provider_errors,
      global_set=True)

  # This ensures updating model cache manager instead of default model cache
  # manager.
  if _MODEL_CACHE_MANAGER is not None:
    _MODEL_CACHE_MANAGER.apply_external_state_changes()

  model_cache_manager = _get_model_cache_manager()
  query_cache_manager = _get_query_cache_manager()
  proxdash_connection = _get_proxdash_connection()

  query_cache_manager.apply_external_state_changes()
  proxdash_connection.apply_external_state_changes()

  for model_connector in _MODEL_CONNECTORS.values():
    model_connector.apply_external_state_changes()

  cache_options = _get_cache_options()
  if cache_options.clear_model_cache_on_connect:
    model_cache_manager.clear_cache()
  if cache_options.clear_query_cache_on_connect:
    query_cache_manager.clear_cache()


def generate_text(
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    messages: Optional[types.MessagesType] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[types.StopType] = None,
    response_format: Optional[types.UserDefinedResponseFormatValueType] = None,
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    use_cache: Optional[bool] = None,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False,
    suppress_provider_errors: Optional[bool] = None) -> Union[str, types.LoggingRecord]:
  if prompt is not None and messages is not None:
    raise ValueError('prompt and messages cannot be set at the same time.')
  if messages is not None:
    type_utils.check_messages_type(messages)

  if use_cache:
    query_cache_manager = _get_query_cache_manager()
    if query_cache_manager.status != types.QueryCacheManagerStatus.WORKING:
      raise ValueError(
          'use_cache is True but query cache is not working.\n'
          f'Query Cache Manager Status: {query_cache_manager.status}')
  elif use_cache is None:
    query_cache_manager = _get_query_cache_manager()
    use_cache = (
        query_cache_manager.status == types.QueryCacheManagerStatus.WORKING)

  if provider_model is not None:
    model_connector = _get_model_connector(
        provider_model_identifier=provider_model)
  else:
    model_connector = _get_registered_model_connector(
        call_type=types.CallType.GENERATE_TEXT)

  response_format: types.ResponseFormat = type_utils.create_response_format(
      response_format)

  logging_record: types.LoggingRecord = model_connector.generate_text(
      prompt=prompt,
      system=system,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop,
      response_format=response_format,
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
  global _AVAILABLE_MODELS
  if _AVAILABLE_MODELS is None:
    _AVAILABLE_MODELS = available_models.AvailableModels(
        get_run_type=_get_run_type,
        get_model_configs=_get_model_configs,
        get_model_connector=_get_model_connector,
        get_allow_multiprocessing=_get_allow_multiprocessing,
        get_model_test_timeout=_get_model_test_timeout,
        get_logging_options=_get_logging_options,
        get_model_cache_manager=_get_model_cache_manager,
        get_proxdash_connection=_get_proxdash_connection)
  return _AVAILABLE_MODELS


def get_current_options(
    json: bool = False) -> Union[types.RunOptions, Dict[str, Any]]:
  run_options = types.RunOptions(
      run_type=_get_run_type(),
      hidden_run_key=_get_hidden_run_key(),
      experiment_path=_get_experiment_path(),
      root_logging_path=_ROOT_LOGGING_PATH,
      default_model_cache_path=_DEFAULT_MODEL_CACHE_PATH,
      logging_options=_get_logging_options(),
      cache_options=_get_cache_options(),
      proxdash_options=_get_proxdash_options(),
      strict_feature_test=_get_strict_feature_test(),
      suppress_provider_errors=_get_suppress_provider_errors(),
      allow_multiprocessing=_get_allow_multiprocessing(),
      model_test_timeout=_get_model_test_timeout())
  if json:
    return type_serializer.encode_run_options(run_options=run_options)
  return run_options


def reset_platform_cache():
  if _PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE and _DEFAULT_MODEL_CACHE_MANAGER:
    _DEFAULT_MODEL_CACHE_MANAGER.clear_cache()


def reset_state():
  reset_platform_cache()
  _init_globals()


def check_health(
    experiment_path: Optional[str]=None,
    verbose: bool = True,
    allow_multiprocessing: bool = True,
    model_test_timeout: int = 25,
    extensive_return: bool = False,
) -> types.ModelStatus:
  if experiment_path is None:
    if _get_experiment_path() is None:
      now = datetime.datetime.now()
      experiment_path = (
          f'connection_health/{now.strftime("%Y-%m-%d_%H-%M-%S")}')
      experiment_path = _set_experiment_path(experiment_path=experiment_path)
      logging_options, _ = _set_logging_options(
          experiment_path=experiment_path,
          logging_options=_get_logging_options())
    else:
      experiment_path = _get_experiment_path()
      logging_options = _get_logging_options()
  if _get_run_type() == types.RunType.TEST:
    proxdash_options = types.ProxDashOptions(
        stdout=False,
        disable_proxdash=True)
  else:
    proxdash_options = copy.deepcopy(_get_proxdash_options())
    proxdash_options.stdout = verbose

  model_configs_instance = _get_model_configs()

  proxdash_connection = proxdash.ProxDashConnection(
      hidden_run_key=_get_hidden_run_key(),
      experiment_path=experiment_path,
      logging_options=logging_options,
      proxdash_options=proxdash_options)

  def _get_modified_model_connector(
      provider_model_identifier: types.ProviderModelIdentifierType
  ) -> model_connector.ProviderModelConnector:
    provider_model = model_configs_instance.get_provider_model(
        provider_model_identifier)
    connector = model_registry.get_model_connector(
        provider_model,
        model_configs=_get_model_configs())
    return connector(
        get_run_type=_get_run_type,
        get_strict_feature_test=_get_strict_feature_test,
        get_query_cache_manager=_get_query_cache_manager,
        logging_options=logging_options,
        proxdash_connection=proxdash_connection,
        stats=_get_stats())

  allow_multiprocessing = _set_allow_multiprocessing(
      allow_multiprocessing=allow_multiprocessing)
  model_test_timeout = _set_model_test_timeout(
      model_test_timeout=model_test_timeout)
  if verbose:
    print('> Starting to test each model...')
  models = available_models.AvailableModels(
      run_type=_get_run_type(),
      model_configs=_get_model_configs(),
      logging_options=logging_options,
      proxdash_connection=proxdash_connection,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
      get_model_connector=_get_modified_model_connector)
  model_status = models.list_models(
      verbose=verbose, return_all=True)
  if verbose:
    providers = set(
        [model.provider for model in model_status.working_models] +
        [model.provider for model in model_status.failed_models])
    result_table = {
        provider: {'working': [], 'failed': []} for provider in providers}
    for model in model_status.working_models:
      result_table[model.provider]['working'].append(model.model)
    for model in model_status.failed_models:
      result_table[model.provider]['failed'].append(model.model)
    print('> Finished testing.\n'
          f'   Registered Providers: {len(providers)}\n'
          f'   Succeeded Models: {len(model_status.working_models)}\n'
          f'   Failed Models: {len(model_status.failed_models)}')
    for provider in sorted(providers):
      print(f'> {provider}:')
      for model in sorted(result_table[provider]['working']):
        provider_model = model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ WORKING | {duration.total_seconds():6.2f}s ]: {model}')
      for model in sorted(result_table[provider]['failed']):
        provider_model = model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ FAILED  | {duration.total_seconds():6.2f}s ]: {model}')
  if proxdash_connection.status == types.ProxDashConnectionStatus.CONNECTED:
    logging_utils.log_proxdash_message(
        logging_options=logging_options,
        proxdash_options=proxdash_options,
        message='Results are uploaded to the ProxDash.',
        type=types.LoggingType.INFO)
  if extensive_return:
    return model_status

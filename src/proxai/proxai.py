import datetime
import traceback
import functools
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
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
import proxai.caching.model_cache as model_cache
import proxai.caching.query_cache as query_cache
import multiprocessing

_RUN_TYPE: types.RunType = types.RunType.PRODUCTION
_REGISTERED_VALUES: Dict[str, types.ModelType] = {}
_INITIALIZED_MODEL_CONNECTORS: Dict[types.ModelType, ModelConnector] = {}
_LOGGING_OPTIONS: types.LoggingOptions = types.LoggingOptions()
_CACHE_OPTIONS: types.CacheOptions = types.CacheOptions()
_QUERY_CACHE_MANAGER: Optional[query_cache.QueryCacheManager] = None
_STRICT_FEATURE_TEST: bool = False

CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions


def _init_globals():
  global _REGISTERED_VALUES
  global _INITIALIZED_MODEL_CONNECTORS
  global _LOGGING_OPTIONS
  global _CACHE_OPTIONS
  global _QUERY_CACHE_MANAGER
  global _STRICT_FEATURE_TEST
  _REGISTERED_VALUES = {}
  _INITIALIZED_MODEL_CONNECTORS = {}
  _LOGGING_OPTIONS = types.LoggingOptions()
  _CACHE_OPTIONS = types.CacheOptions()
  _QUERY_CACHE_MANAGER = None
  _STRICT_FEATURE_TEST = False


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
      strict_feature_test=_STRICT_FEATURE_TEST)

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


class AvailableModels:
  _model_cache: Optional[model_cache.ModelCache] = None
  _generate_text: Dict[types.ModelType, Any] = {}
  _providers_with_key: Set[types.Provider] = set()

  def __init__(self):
    self._load_provider_keys()

  def _load_provider_keys(self):
    for provider, provider_key_name in types.PROVIDER_KEY_MAP.items():
      provider_flag = True
      for key_name in provider_key_name:
        if key_name not in os.environ:
          provider_flag = False
          break
      if provider_flag:
        self._providers_with_key.add(provider)

  def generate_text(
      self,
      only_largest_models: bool = False,
      verbose: bool = False,
      failed_models: bool = False
  ) -> List[types.ModelType]:
    start_time = datetime.datetime.now()
    models = types.ModelStatus()
    self._get_all_models(models, call_type=types.CallType.GENERATE_TEXT)
    self._filter_by_provider_key(models)
    self._filter_by_cache(models, call_type=types.CallType.GENERATE_TEXT)

    # TODO: This very experimental and require proper design. One alternative is
    # registering models according to their sizes in px.types. Then, find
    # working largest model for each provider.
    if only_largest_models:
      _allowed_models = set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS),
          (types.Provider.GEMINI, types.GeminiModel.GEMINI_1_5_PRO_LATEST),
          (types.Provider.COHERE, types.CohereModel.COMMAND_R),
          (types.Provider.DATABRICKS,
           types.DatabricksModel.DATABRICKS_DBRX_INSTRUCT),
          (types.Provider.DATABRICKS,
           types.DatabricksModel.DATABRICKS_LLAMA_2_70b_CHAT),
          (types.Provider.MISTRAL, types.MistralModel.MISTRAL_LARGE_LATEST),
      ])
      for model in list(models.unprocessed_models):
        if model not in _allowed_models:
          models.unprocessed_models.remove(model)
          models.filtered_models.add(model)
      for model in list(models.working_models):
        if model not in _allowed_models:
          models.working_models.remove(model)
          models.filtered_models.add(model)

    print_flag = bool(verbose and models.unprocessed_models)
    if print_flag:
      print(f'From cache;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      print(f'Running test for {len(models.unprocessed_models)} models.')
    self._test_models(models, call_type=types.CallType.GENERATE_TEXT)
    end_time = datetime.datetime.now()
    if print_flag:
      print(f'After test;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      duration = (end_time - start_time).total_seconds()
      print(f'Test duration: {duration} seconds.')

    if failed_models:
      return self._format_set(models.failed_models)
    return self._format_set(models.working_models)

  def _get_all_models(self, models: types.ModelStatus, call_type: str):
    if call_type == types.CallType.GENERATE_TEXT:
      for provider, provider_models in types.GENERATE_TEXT_MODELS.items():
        for provider_model in provider_models:
          if provider_model not in (
              models.working_models
              | models.failed_models
              | models.filtered_models):
            models.unprocessed_models.add((provider, provider_model))

  def _filter_by_provider_key(self, models: types.ModelStatus):
    def _filter_set(
        model_set: Set[types.ModelType]
    ) -> Tuple[Set[types.ModelType], Set[types.ModelType]]:
      not_filtered = set()
      for model in model_set:
        provider, _ = model
        if provider in self._providers_with_key:
          not_filtered.add(model)
        else:
          models.filtered_models.add(model)
      return not_filtered
    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_cache(
      self,
      models: types.ModelStatus,
      call_type: str):
    if not _CACHE_OPTIONS.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(_CACHE_OPTIONS)
    cache_result = self._model_cache.get(call_type=call_type)

    def _remove_model(model: types.ModelType):
      if model in models.unprocessed_models:
        models.unprocessed_models.remove(model)
      if model in models.working_models:
        models.working_models.remove(model)
      elif model in models.failed_models:
        models.failed_models.remove(model)

    for model in cache_result.working_models:
      if model not in models.filtered_models:
        _remove_model(model)
        models.working_models.add(model)
    for model in cache_result.failed_models:
      if model not in models.filtered_models:
        _remove_model(model)
        models.failed_models.add(model)

  @staticmethod
  def _test_generate_text(
      model: types.ModelType,
      model_connector: ModelConnector
  ) -> List[types.LoggingRecord]:
    start_time = datetime.datetime.now()
    prompt = 'Hello model!?'
    max_tokens = 100
    try:
      logging_record = model_connector.generate_text(
          model=model,
          prompt=prompt,
          max_tokens=max_tokens,
          use_cache=False)
      return logging_record
    except Exception as e:
      return types.LoggingRecord(
          query_record=types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT,
              model=model,
              prompt=prompt,
              max_tokens=max_tokens),
          response_record=types.QueryResponseRecord(
              error=str(e),
              error_traceback=traceback.format_exc(),
              start_time=start_time,
              end_time=datetime.datetime.now(),
              response_time=datetime.datetime.now() - start_time),
          response_source=types.ResponseSource.PROVIDER)

  def _test_models(self, models: types.ModelStatus, call_type: str):
    global _INITIALIZED_MODEL_CONNECTORS
    if not models.unprocessed_models:
      return
    for model in models.unprocessed_models:
      if model not in _INITIALIZED_MODEL_CONNECTORS:
        _INITIALIZED_MODEL_CONNECTORS[model] = _init_model_connector(model)

    test_models = list(models.unprocessed_models)
    test_func = None
    if call_type == types.CallType.GENERATE_TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(f'Call type not supported: {call_type}')
    pool = multiprocessing.Pool(processes=len(test_models))
    test_results = []
    for test_model in test_models:
      result = pool.apply_async(
          test_func,
          args=(test_model, _INITIALIZED_MODEL_CONNECTORS[test_model]))
      test_results.append(result)
    pool.close()
    pool.join()
    test_results: List[types.LoggingRecord] = [
        result.get() for result in test_results]
    update_models = types.ModelStatus()
    for logging_record in test_results:
      models.unprocessed_models.remove(logging_record.query_record.model)
      if logging_record.response_record.response != None:
        models.working_models.add(logging_record.query_record.model)
        update_models.working_models.add(logging_record.query_record.model)
      else:
        models.failed_models.add(logging_record.query_record.model)
        update_models.failed_models.add(logging_record.query_record.model)
      update_models.provider_queries.append(logging_record)
    if not _CACHE_OPTIONS.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(_CACHE_OPTIONS)
    self._model_cache.update(model_status=update_models, call_type=call_type)

  def _format_set(
      self,
      model_set: Set[types.ModelType]
  ) -> List[types.ModelType]:
    return sorted(list(model_set))

import copy
import datetime
import multiprocessing
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import proxai.types as types
import proxai.caching.model_cache as model_cache
from proxai.connectors.model_connector import ModelConnector
from proxai.connections.proxdash import ProxDashConnection


class AvailableModels:
  _model_cache: Optional[model_cache.ModelCache] = None
  _generate_text: Dict[types.ModelType, Any] = {}
  _run_type: types.RunType
  _get_run_type: Callable[[], types.RunType]
  _cache_options: types.CacheOptions
  _get_cache_options: Callable[[], types.CacheOptions]
  _logging_options: types.LoggingOptions
  _get_logging_options: Callable[[], types.LoggingOptions]
  _proxdash_connection: ProxDashConnection
  _get_proxdash_connection: Callable[[], ProxDashConnection]
  _get_initialized_model_connectors: Callable[
      [], Dict[types.ModelType, ModelConnector]]
  _init_model_connector: Callable[[types.ModelType], ModelConnector]
  _providers_with_key: Set[types.Provider] = set()

  def __init__(
      self,
      get_initialized_model_connectors: Callable[
          [], Dict[types.ModelType, ModelConnector]],
      init_model_connector: Callable[[types.ModelType], ModelConnector],
      run_type: types.RunType = None,
      get_run_type: Callable[[], types.RunType] = None,
      cache_options: Optional[types.CacheOptions] = None,
      get_cache_options: Optional[Callable[[], types.CacheOptions]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[], ProxDashConnection]] = None):
    if run_type and get_run_type:
      raise ValueError(
          'Only one of run_type or get_run_type should be provided.')
    if cache_options and get_cache_options:
      raise ValueError(
          'Only one of cache_options or get_cache_options should be provided.')
    if logging_options and get_logging_options:
      raise ValueError(
          'Only one of logging_options or get_logging_options should be '
          'provided.')
    if proxdash_connection and get_proxdash_connection:
      raise ValueError(
          'Only one of proxdash_connection or get_proxdash_connection should '
          'be provided.')
    self.run_type = run_type
    self._get_run_type = get_run_type
    self.cache_options = cache_options
    self._get_cache_options = get_cache_options
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    self.proxdash_connection = proxdash_connection
    self._get_proxdash_connection = get_proxdash_connection
    self._get_initialized_model_connectors = get_initialized_model_connectors
    self._init_model_connector= init_model_connector
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

  @property
  def run_type(self) -> types.RunType:
    if self._run_type:
      return self._run_type
    if self._get_run_type:
      return self._get_run_type()
    return None

  @run_type.setter
  def run_type(self, run_type: types.RunType):
    self._run_type = run_type

  @property
  def cache_options(self) -> types.CacheOptions:
    if self._cache_options:
      return self._cache_options
    if self._get_cache_options:
      return self._get_cache_options()
    return None

  @cache_options.setter
  def cache_options(self, cache_options: types.CacheOptions):
    self._cache_options = cache_options

  @property
  def logging_options(self) -> types.LoggingOptions:
    if self._logging_options:
      return self._logging_options
    if self._get_logging_options:
      return self._get_logging_options()
    return None

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self._logging_options = logging_options

  @property
  def proxdash_connection(self) -> ProxDashConnection:
    if self.run_type == types.RunType.TEST:
      return None
    if self._proxdash_connection:
      return self._proxdash_connection
    if self._get_proxdash_connection:
      return self._get_proxdash_connection()
    return None

  @proxdash_connection.setter
  def proxdash_connection(self, proxdash_connection: ProxDashConnection):
    self._proxdash_connection = proxdash_connection

  def generate_text(
      self,
      only_largest_models: bool = False,
      verbose: bool = False,
      return_all: bool = False
  ) -> List[types.ModelType]:
    start_time = datetime.datetime.now()
    models = types.ModelStatus()
    self._get_all_models(models, call_type=types.CallType.GENERATE_TEXT)
    self._filter_by_provider_key(models)
    self._filter_by_cache(models, call_type=types.CallType.GENERATE_TEXT)

    # TODO: This is very experimental and require proper design. One alternative
    # is registering models according to their sizes in px.types. Then, find
    # working largest model for each provider.
    if only_largest_models:
      _allowed_models = set([
          (types.Provider.OPENAI, types.OpenAIModel.GPT_4_TURBO_PREVIEW),
          (types.Provider.CLAUDE, types.ClaudeModel.CLAUDE_3_OPUS),
          (types.Provider.GEMINI, types.GeminiModel.GEMINI_1_5_PRO_LATEST),
          (types.Provider.COHERE, types.CohereModel.COMMAND_R_PLUS),
          (types.Provider.DATABRICKS, types.DatabricksModel.DBRX_INSTRUCT),
          (types.Provider.DATABRICKS,
           types.DatabricksModel.LLAMA_3_70B_INSTRUCT),
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

    if return_all:
      return (
          self._format_set(models.working_models),
          self._format_set(models.failed_models))
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
    if not self.cache_options.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(self.cache_options)
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
      model_connector: ModelConnector,
      logging_options: types.LoggingOptions,
      proxdash_connection: Optional[ProxDashConnection],
  ) -> List[types.LoggingRecord]:
    start_time = datetime.datetime.now()
    prompt = 'Hello model!'
    max_tokens = 100
    model_connector = copy.deepcopy(model_connector)
    model_connector._logging_options = logging_options
    model_connector._get_logging_options = None
    model_connector._proxdash_connection = None
    model_connector._get_proxdash_connection = None
    if proxdash_connection:
      model_connector._proxdash_connection = proxdash_connection
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
    if not models.unprocessed_models:
      return
    initialized_model_connectors = self._get_initialized_model_connectors()
    for model in models.unprocessed_models:
      if model not in initialized_model_connectors:
        initialized_model_connectors[
            model] = self._init_model_connector(model)

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
          args=(
              test_model,
              initialized_model_connectors[test_model],
              self.logging_options,
              self.proxdash_connection))
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
    if not self.cache_options.path:
      return
    if not self._model_cache:
      self._model_cache = model_cache.ModelCache(self.cache_options)
    self._model_cache.update(model_status=update_models, call_type=call_type)

  def _format_set(
      self,
      model_set: Set[types.ModelType]
  ) -> List[types.ModelType]:
    return sorted(list(model_set))

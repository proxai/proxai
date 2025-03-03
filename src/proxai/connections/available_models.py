import copy
import datetime
import multiprocessing
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import proxai.types as types
import proxai.caching.model_cache as model_cache
import proxai.connectors.model_registry as model_registry
from proxai.connectors.model_connector import ProviderModelConnector
from proxai.connections.proxdash import ProxDashConnection
import proxai.connectors.model_configs as model_configs


class AvailableModels:
  _model_cache_manager: Optional[model_cache.ModelCacheManager]
  _generate_text: Dict[types.ProviderModelType, Any]
  _run_type: types.RunType
  _get_run_type: Callable[[], types.RunType]
  _cache_options: types.CacheOptions
  _get_cache_options: Callable[[], types.CacheOptions]
  _logging_options: types.LoggingOptions
  _get_logging_options: Callable[[], types.LoggingOptions]
  _allow_multiprocessing: bool
  _get_allow_multiprocessing: Callable[[], bool]
  _proxdash_connection: ProxDashConnection
  _get_proxdash_connection: Callable[[], ProxDashConnection]
  _get_initialized_model_connectors: Callable[
      [], Dict[types.ProviderModelType, ProviderModelConnector]]
  _init_model_connector: Callable[
      [types.ProviderModelType], ProviderModelConnector]
  _providers_with_key: Set[str]
  _has_fetched_all_models: bool

  def __init__(
      self,
      get_initialized_model_connectors: Callable[
          [], Dict[types.ProviderModelType, ProviderModelConnector]],
      init_model_connector: Callable[
          [types.ProviderModelType], ProviderModelConnector],
      run_type: types.RunType = None,
      get_run_type: Callable[[], types.RunType] = None,
      model_cache_manager: Optional[model_cache.ModelCacheManager] = None,
      get_model_cache_manager: Optional[
          Callable[[], model_cache.ModelCacheManager]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[], ProxDashConnection]] = None,
      allow_multiprocessing: bool = None,
      get_allow_multiprocessing: Optional[Callable[[], bool]] = None):
    if run_type and get_run_type:
      raise ValueError(
          'Only one of run_type or get_run_type should be provided.')
    if logging_options and get_logging_options:
      raise ValueError(
          'Only one of logging_options or get_logging_options should be '
          'provided.')
    if proxdash_connection and get_proxdash_connection:
      raise ValueError(
          'Only one of proxdash_connection or get_proxdash_connection should '
          'be provided.')
    if allow_multiprocessing and get_allow_multiprocessing:
      raise ValueError(
          'Only one of allow_multiprocessing or get_allow_multiprocessing should '
          'be provided.')
    if model_cache_manager and get_model_cache_manager:
      raise ValueError(
          'Only one of model_cache_manager or get_model_cache_manager should '
          'be provided.')
    self.run_type = run_type
    self._generate_text = {}
    self._get_run_type = get_run_type
    self.model_cache_manager = model_cache_manager
    self._get_model_cache_manager = get_model_cache_manager
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    self.proxdash_connection = proxdash_connection
    self._get_proxdash_connection = get_proxdash_connection
    self._get_initialized_model_connectors = get_initialized_model_connectors
    self._init_model_connector= init_model_connector
    self.allow_multiprocessing = allow_multiprocessing
    self._get_allow_multiprocessing = get_allow_multiprocessing
    self._providers_with_key = set()
    self._has_fetched_all_models = False
    self._load_provider_keys()

  def _load_provider_keys(self):
    self._providers_with_key = set()
    for provider, provider_key_name in model_configs.PROVIDER_KEY_MAP.items():
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
  def model_cache_manager(self) -> model_cache.ModelCacheManager:
    if self._model_cache_manager:
      return self._model_cache_manager
    if self._get_model_cache_manager:
      return self._get_model_cache_manager()
    return None

  @model_cache_manager.setter
  def model_cache_manager(
      self, model_cache_manager: model_cache.ModelCacheManager):
    self._model_cache_manager = model_cache_manager

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

  @property
  def allow_multiprocessing(self) -> bool:
    if self._allow_multiprocessing is not None:
      return self._allow_multiprocessing
    if self._get_allow_multiprocessing:
      return self._get_allow_multiprocessing()
    return None

  @allow_multiprocessing.setter
  def allow_multiprocessing(self, allow_multiprocessing: bool):
    self._allow_multiprocessing = allow_multiprocessing

  def _get_all_models(self, models: types.ModelStatus, call_type: str):
    if call_type == types.CallType.GENERATE_TEXT:
      for provider_models in model_configs.GENERATE_TEXT_MODELS.values():
        for provider_model in provider_models.values():
          if provider_model not in (
              models.working_models
              | models.failed_models
              | models.filtered_models):
            models.unprocessed_models.add(provider_model)
      self._update_provider_queries(models)

  def _filter_by_provider_key(self, models: types.ModelStatus):
    def _filter_set(
        provider_model_set: Set[types.ProviderModelType]
    ) -> Tuple[Set[types.ProviderModelType], Set[types.ProviderModelType]]:
      not_filtered = set()
      for provider_model in provider_model_set:
        if provider_model.provider in self._providers_with_key:
          not_filtered.add(provider_model)
        else:
          models.filtered_models.add(provider_model)
      return not_filtered
    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)
    self._update_provider_queries(models)

  def _filter_by_cache(
      self,
      models: types.ModelStatus,
      call_type: str):
    if not self.model_cache_manager:
      return
    cache_result = types.ModelStatus()
    if self.model_cache_manager:
      cache_result = self.model_cache_manager.get(call_type=call_type)

    def _remove_model(provider_model: types.ProviderModelType):
      if provider_model in models.unprocessed_models:
        models.unprocessed_models.remove(provider_model)
      if provider_model in models.working_models:
        models.working_models.remove(provider_model)
      if provider_model in models.failed_models:
        models.failed_models.remove(provider_model)

    provider_query_map = {
        query.query_record.provider_model: query
        for query in cache_result.provider_queries
    }

    for provider_model in cache_result.working_models:
      if provider_model not in models.filtered_models:
        _remove_model(provider_model)
        models.working_models.add(provider_model)
        models.provider_queries.append(provider_query_map[provider_model])
    for provider_model in cache_result.failed_models:
      if provider_model not in models.filtered_models:
        _remove_model(provider_model)
        models.failed_models.add(provider_model)
        models.provider_queries.append(provider_query_map[provider_model])
    self._update_provider_queries(models)

  def _filter_largest_models(self, models: types.ModelStatus):
    # TODO: This is very experimental and require proper design. One alternative
    # is registering models according to their sizes in px.types. Then, find
    # working largest model for each provider.

    _allowed_models = set([
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview'],
        model_configs.ALL_MODELS['claude']['claude-3-opus'],
        model_configs.ALL_MODELS['gemini']['gemini-1.5-pro-latest'],
        model_configs.ALL_MODELS['cohere']['command-r-plus'],
        model_configs.ALL_MODELS['databricks']['dbrx-instruct'],
        model_configs.ALL_MODELS['databricks']['llama-3-70b-instruct'],
        model_configs.ALL_MODELS['mistral']['mistral-large-latest'],
    ])
    for model in list(models.unprocessed_models):
      if model not in _allowed_models:
        models.unprocessed_models.remove(model)
        models.filtered_models.add(model)
    for model in list(models.working_models):
      if model not in _allowed_models:
        models.working_models.remove(model)
        models.filtered_models.add(model)

    self._update_provider_queries(models)

  @staticmethod
  def _test_generate_text(
      model_init_state: types.ModelInitState,
  ) -> List[types.LoggingRecord]:
    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    prompt = 'Hello model!'
    max_tokens = 100
    model_connector = model_registry.get_model_connector(
        model_init_state.provider_model)
    model_connector = model_connector(init_state=model_init_state)

    try:
      logging_record: types.LoggingRecord = model_connector.generate_text(
          prompt=prompt,
          max_tokens=max_tokens,
          use_cache=False)
      return logging_record
    except Exception as e:
      return types.LoggingRecord(
          query_record=types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT,
              provider_model=model_init_state.provider_model,
              prompt=prompt,
              max_tokens=max_tokens),
          response_record=types.QueryResponseRecord(
              error=str(e),
              error_traceback=traceback.format_exc(),
              start_utc_date=start_utc_date,
              end_utc_date=datetime.datetime.now(datetime.timezone.utc),
              response_time=(
                  datetime.datetime.now(datetime.timezone.utc)
                  - start_utc_date)),
          response_source=types.ResponseSource.PROVIDER)

  def _test_models(self, models: types.ModelStatus, call_type: str):
    if not models.unprocessed_models:
      return

    initialized_model_connectors = self._get_initialized_model_connectors()
    for provider_model in models.unprocessed_models:
      if provider_model not in initialized_model_connectors:
        initialized_model_connectors[
            provider_model] = self._init_model_connector(provider_model)

    test_provider_models = list(models.unprocessed_models)
    test_func = None
    if call_type == types.CallType.GENERATE_TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(f'Call type not supported: {call_type}')

    test_results = []
    if self.allow_multiprocessing:
      pool = multiprocessing.Pool(processes=len(test_provider_models))
      for test_provider_model in test_provider_models:
        result = pool.apply_async(
            test_func,
            args=(initialized_model_connectors[test_provider_model].get_init_state(),))
        test_results.append(result)
      pool.close()
      pool.join()
      test_results: List[types.LoggingRecord] = [
          result.get() for result in test_results]
    else:
      for test_provider_model in test_provider_models:
        test_results.append(
            test_func(
                initialized_model_connectors[test_provider_model].get_init_state()))

    update_models = types.ModelStatus()
    for logging_record in test_results:
      models.unprocessed_models.remove(logging_record.query_record.provider_model)
      if logging_record.response_record.response != None:
        models.working_models.add(logging_record.query_record.provider_model)
        update_models.working_models.add(logging_record.query_record.provider_model)
      else:
        models.failed_models.add(logging_record.query_record.provider_model)
        update_models.failed_models.add(logging_record.query_record.provider_model)
      models.provider_queries.append(logging_record)
      update_models.provider_queries.append(logging_record)
    if self.model_cache_manager:
      self.model_cache_manager.update(
          model_status=update_models, call_type=call_type)
    self._update_provider_queries(models)

  def _format_set(
      self,
      provider_model_set: Set[types.ProviderModelType]
  ) -> List[types.ProviderModelType]:
    return sorted(list(provider_model_set))

  def _update_provider_queries(
      self,
      models: types.ModelStatus):
    models.provider_queries = [
        query for query in models.provider_queries
        if query.query_record.provider_model in models.working_models or
        query.query_record.provider_model in models.failed_models
    ]

  def get_all_models(
      self,
      only_largest_models: bool = False,
      verbose: bool = False,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> Union[Set[types.ProviderModelType], types.ModelStatus]:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    models = types.ModelStatus()
    self._load_provider_keys()
    self._get_all_models(models, call_type=call_type)
    self._filter_by_provider_key(models)
    if clear_model_cache and self.model_cache_manager:
      self.model_cache_manager.clear_cache()
    self._filter_by_cache(models, call_type=types.CallType.GENERATE_TEXT)
    if only_largest_models:
      self._filter_largest_models(models)

    print_flag = bool(verbose and models.unprocessed_models)
    if print_flag:
      print(f'From cache;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      print(f'Running test for {len(models.unprocessed_models)} models.')
    self._test_models(models, call_type=types.CallType.GENERATE_TEXT)
    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    if print_flag:
      print(f'After test;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      duration = (end_utc_date - start_utc_date).total_seconds()
      print(f'Test duration: {duration} seconds.')

    if not only_largest_models:
      self._has_fetched_all_models = True

    if return_all:
      return models
    return self._format_set(models.working_models)

  def get_providers(
      self,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> List[str]:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    if not self.model_cache_manager:
      self._load_provider_keys()
      return sorted(list(self._providers_with_key))

    if clear_model_cache:
      self.model_cache_manager.clear_cache()

    if clear_model_cache or not self._has_fetched_all_models:
      self.get_all_models(
          only_largest_models=False,
          verbose=verbose,
          call_type=call_type)

    cached_results = self.model_cache_manager.get(call_type=call_type)
    providers = set([
        model.provider
        for model in cached_results.working_models])
    return sorted(list(providers))

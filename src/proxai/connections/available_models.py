import copy
import datetime
import multiprocessing
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import proxai.types as types
import proxai.caching.model_cache as model_cache
import proxai.connections.proxdash as proxdash
import proxai.connectors.model_registry as model_registry
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_configs as model_configs
import proxai.logging.utils as logging_utils
import proxai.state_controllers.state_controller as state_controller
import concurrent.futures

_AVAILABLE_MODELS_STATE_PROPERTY = '_available_models_state'
_GENERATE_TEXT_TEST_PROMPT = 'Hello model!'
_GENERATE_TEXT_TEST_MAX_TOKENS = 1000


class AvailableModels(state_controller.StateControlled):
  _model_cache_manager: Optional[model_cache.ModelCacheManager]
  _run_type: types.RunType
  _get_run_type: Callable[[], types.RunType]
  _get_model_connector: Callable[
      [types.ProviderModelType], model_connector.ProviderModelConnector]
  _cache_options: types.CacheOptions
  _get_cache_options: Callable[[], types.CacheOptions]
  _logging_options: types.LoggingOptions
  _get_logging_options: Callable[[], types.LoggingOptions]
  _allow_multiprocessing: bool
  _get_allow_multiprocessing: Callable[[], bool]
  _model_test_timeout: int
  _get_model_test_timeout: Callable[[], int]
  _proxdash_connection: proxdash.ProxDashConnection
  _get_proxdash_connection: Callable[[], proxdash.ProxDashConnection]
  _providers_with_key: Set[str]
  _has_fetched_all_models: bool
  _latest_model_cache_path_used_for_update: Optional[str]
  _available_models_state: types.AvailableModelsState

  def __init__(
      self,
      run_type: Optional[types.RunType] = None,
      get_run_type: Optional[Callable[[], types.RunType]] = None,
      model_cache_manager: Optional[model_cache.ModelCacheManager] = None,
      get_model_cache_manager: Optional[
          Callable[[], model_cache.ModelCacheManager]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_connection: Optional[proxdash.ProxDashConnection] = None,
      get_proxdash_connection: Optional[
          Callable[[], proxdash.ProxDashConnection]] = None,
      allow_multiprocessing: bool = None,
      get_allow_multiprocessing: Optional[Callable[[], bool]] = None,
      model_test_timeout: Optional[int] = None,
      get_model_test_timeout: Optional[Callable[[], int]] = None,
      get_model_connector: Optional[
          Callable[[], model_connector.ProviderModelConnector]] = None,
      init_state: Optional[types.AvailableModelsState] = None):

    if init_state and (
        run_type is not None or
        get_run_type is not None or
        model_cache_manager is not None or
        get_model_cache_manager is not None or
        logging_options is not None or
        get_logging_options is not None or
        proxdash_connection is not None or
        get_proxdash_connection is not None or
        allow_multiprocessing is not None or
        get_allow_multiprocessing is not None or
        model_test_timeout is not None or
        get_model_test_timeout is not None):
      raise ValueError(
          'init_state and other parameters cannot be set at the same time.')

    super().__init__(
        run_type=run_type,
        get_run_type=get_run_type,
        model_cache_manager=model_cache_manager,
        get_model_cache_manager=get_model_cache_manager,
        logging_options=logging_options,
        get_logging_options=get_logging_options,
        proxdash_connection=proxdash_connection,
        get_proxdash_connection=get_proxdash_connection,
        allow_multiprocessing=allow_multiprocessing,
        get_allow_multiprocessing=get_allow_multiprocessing,
        model_test_timeout=model_test_timeout,
        get_model_test_timeout=get_model_test_timeout)

    self.init_state()

    if init_state:
      self.load_state(init_state)
    else:
      initial_state = self.get_state()
      self._get_run_type = get_run_type
      self._get_model_cache_manager = get_model_cache_manager
      self._get_logging_options = get_logging_options
      self._get_proxdash_connection = get_proxdash_connection
      self._get_allow_multiprocessing = get_allow_multiprocessing
      self._get_model_test_timeout = get_model_test_timeout
      self._get_model_connector = get_model_connector

      self.run_type = run_type
      self.model_cache_manager = model_cache_manager
      self.logging_options = logging_options
      self.proxdash_connection = proxdash_connection
      self.allow_multiprocessing = allow_multiprocessing
      self.model_test_timeout = model_test_timeout
      self.get_model_connector = get_model_connector

      self.providers_with_key = set()
      self.has_fetched_all_models = False
      self.latest_model_cache_path_used_for_update = None
      self._load_provider_keys()

      self.handle_changes(initial_state, self.get_state())

  def get_internal_state_property_name(self):
    return _AVAILABLE_MODELS_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.AvailableModelsState

  def handle_changes(
      self,
      old_state: types.AvailableModelsState,
      current_state: types.AvailableModelsState):
    pass

  def _load_provider_keys(self):
    self.providers_with_key = set()
    for provider, provider_key_name in model_configs.PROVIDER_KEY_MAP.items():
      provider_flag = True
      for key_name in provider_key_name:
        if key_name not in os.environ:
          provider_flag = False
          break
      if provider_flag:
        self.providers_with_key.add(provider)

  @property
  def run_type(self) -> types.RunType:
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, run_type: types.RunType):
    self.set_property_value('run_type', run_type)

  @property
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self.set_property_value('logging_options', logging_options)

  @property
  def model_cache_manager(self) -> model_cache.ModelCacheManager:
    return self.get_state_controlled_property_value('model_cache_manager')

  @model_cache_manager.setter
  def model_cache_manager(
      self, value: model_cache.ModelCacheManager):
    self.set_state_controlled_property_value('model_cache_manager', value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value: proxdash.ProxDashConnection):
    self.set_state_controlled_property_value('proxdash_connection', value)

  @property
  def allow_multiprocessing(self) -> bool:
    return self.get_property_value('allow_multiprocessing')

  @allow_multiprocessing.setter
  def allow_multiprocessing(self, value: bool):
    self.set_property_value('allow_multiprocessing', value)

  @property
  def model_test_timeout(self) -> int:
    return self.get_property_value('model_test_timeout')

  @model_test_timeout.setter
  def model_test_timeout(self, value: int):
    self.set_property_value('model_test_timeout', value)

  @property
  def providers_with_key(self) -> Set[str]:
    return self.get_property_value('providers_with_key')

  @providers_with_key.setter
  def providers_with_key(self, value: Set[str]):
    self.set_property_value('providers_with_key', value)

  @property
  def has_fetched_all_models(self) -> bool:
    return self.get_property_value('has_fetched_all_models')

  @has_fetched_all_models.setter
  def has_fetched_all_models(self, value: bool):
    self.set_property_value('has_fetched_all_models', value)

  @property
  def latest_model_cache_path_used_for_update(self) -> Optional[str]:
    return self.get_property_value('latest_model_cache_path_used_for_update')

  @latest_model_cache_path_used_for_update.setter
  def latest_model_cache_path_used_for_update(
      self, value: Optional[str]):
    self.set_property_value('latest_model_cache_path_used_for_update', value)

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
        if provider_model.provider in self.providers_with_key:
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

    for provider_model in cache_result.working_models:
      if provider_model not in models.filtered_models:
        _remove_model(provider_model)
        models.working_models.add(provider_model)
        models.provider_queries[
            provider_model] = cache_result.provider_queries[provider_model]
    for provider_model in cache_result.failed_models:
      if provider_model not in models.filtered_models:
        _remove_model(provider_model)
        models.failed_models.add(provider_model)
        models.provider_queries[
            provider_model] = cache_result.provider_queries[provider_model]
    self._update_provider_queries(models)

  def _filter_largest_models(self, models: types.ModelStatus):
    _allowed_models = set()
    for provider_models in model_configs.LARGEST_GENERATE_TEXT_MODELS.values():
      for provider_model in provider_models.values():
        _allowed_models.add(provider_model)

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
      provider_model_state: types.ProviderModelState,
      verbose: bool = False
  ) -> List[types.LoggingRecord]:
    if verbose:
      print(f'Testing {provider_model_state.provider_model}...')
    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    model_connector = model_registry.get_model_connector(
        provider_model_state.provider_model,
        without_additional_args=True)
    model_connector = model_connector(init_state=provider_model_state)
    try:
      logging_record: types.LoggingRecord = model_connector.generate_text(
          prompt=_GENERATE_TEXT_TEST_PROMPT,
          max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS,
          use_cache=False)
      return logging_record
    except Exception as e:
      return types.LoggingRecord(
          query_record=types.QueryRecord(
              call_type=types.CallType.GENERATE_TEXT,
              provider_model=provider_model_state.provider_model,
              prompt=_GENERATE_TEXT_TEST_PROMPT,
              max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS),
          response_record=types.QueryResponseRecord(
              error=str(e),
              error_traceback=traceback.format_exc(),
              start_utc_date=start_utc_date,
              end_utc_date=datetime.datetime.now(datetime.timezone.utc),
              response_time=(
                  datetime.datetime.now(datetime.timezone.utc)
                  - start_utc_date)),
          response_source=types.ResponseSource.PROVIDER)

  def _get_timeout_logging_record(
      self,
      provider_model: types.ProviderModelType):
    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    start_utc_date = end_utc_date - datetime.timedelta(
        seconds=self.model_test_timeout)
    return types.LoggingRecord(
        query_record=types.QueryRecord(
            call_type=types.CallType.GENERATE_TEXT,
            provider_model=provider_model,
            prompt=_GENERATE_TEXT_TEST_PROMPT,
            max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS),
        response_record=types.QueryResponseRecord(
            error=(
                f'Model {provider_model} took longer than '
                f'{self.model_test_timeout} seconds to respond'),
            error_traceback=traceback.format_exc(),
            start_utc_date=start_utc_date,
            end_utc_date=end_utc_date,
            response_time=(end_utc_date - start_utc_date)),
          response_source=types.ResponseSource.PROVIDER)

  def _get_bootstrap_error(self, e: Exception):
    error_str = str(e).lower()
    is_bootstrapping_error = (
      "an attempt has been made to start a new process before" in error_str and
      "current process has finished its bootstrapping phase" in error_str
    )
    if is_bootstrapping_error:
      return Exception(
          f'{e}\n\nMultiprocessing initialization error: Unable to start new '
          'processes because the proxai library was imported and used '
          'outside of the "if __name__ == \'__main__\':" block. To fix '
          'this:\n'
          '1. Move your proxai code inside a "if __name__ == \'__main__\':"'
          ' block, or\n'
          '2. Disable multiprocessing by setting '
          'allow_multiprocessing=False on px.connect()\n'
          'For more details, see followings:\n'
          '- https://www.proxai.co/proxai-docs/advanced/multiprocessing\n'
          '- https://docs.python.org/3/library/multiprocessing.html#the-'
          'spawn-and-forkserver-start-methods\n')
    return None


  def _test_models_with_multiprocessing(
      self,
      model_connectors: Dict[
        types.ProviderModelType, model_connector.ProviderModelConnector],
      call_type: str,
      verbose: bool = False):
    process_count = max(1, multiprocessing.cpu_count() - 1)
    test_func = None
    if call_type == types.CallType.GENERATE_TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(f'Call type not supported: {call_type}')

    test_results = []
    try:
      pool = multiprocessing.Pool(processes=process_count)
      pool_results = []
      for provider_model, connector in model_connectors.items():
        pool_result = pool.apply_async(
            test_func,
            args=(connector.get_state(), verbose))
        pool_results.append((provider_model, pool_result))
      pool.close()
      for provider_model, pool_result in pool_results:
        try:
          test_results.append(pool_result.get(timeout=self.model_test_timeout))
        except multiprocessing.TimeoutError:
          if verbose:
            print(
                f"> {provider_model} query took longer than "
                f'{self.model_test_timeout} seconds to respond')
          test_results.append(self._get_timeout_logging_record(provider_model))
      pool.terminate()
      pool.join()
    except Exception as e:
      bootstrap_error = self._get_bootstrap_error(e)
      if bootstrap_error:
        raise bootstrap_error
      else:
        raise e
    return test_results

  def _test_models_sequentially(
      self,
      model_connectors: Dict[
        types.ProviderModelType, model_connector.ProviderModelConnector],
      call_type: str,
      verbose: bool = False):
    """Tests provider models sequentially.

    This function is used when multiprocessing is disabled. Note that this
    function cannot handle model timeouts because of the python's limitation
    on timeout handling without multiprocessing/threading.
    """
    test_func = None
    if call_type == types.CallType.GENERATE_TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(f'Call type not supported: {call_type}')

    test_results = []
    for connector in model_connectors.values():
      test_results.append(test_func(connector.get_state(), verbose))

    return test_results

  def _test_models(
      self,
      models: types.ModelStatus,
      call_type: str,
      verbose: bool = False):
    if not models.unprocessed_models:
      return

    model_connectors = {}
    for provider_model in models.unprocessed_models:
      model_connectors[provider_model] = self.get_model_connector(
          provider_model)

    if self.allow_multiprocessing:
      test_results = self._test_models_with_multiprocessing(
          model_connectors=model_connectors,
          call_type=call_type,
          verbose=verbose)
    else:
      test_results = self._test_models_sequentially(
          model_connectors=model_connectors,
          call_type=call_type,
          verbose=verbose)

    update_models = types.ModelStatus()
    for logging_record in test_results:
      models.unprocessed_models.remove(logging_record.query_record.provider_model)
      if logging_record.response_record.response != None:
        models.working_models.add(logging_record.query_record.provider_model)
        update_models.working_models.add(logging_record.query_record.provider_model)
      else:
        models.failed_models.add(logging_record.query_record.provider_model)
        update_models.failed_models.add(logging_record.query_record.provider_model)
      models.provider_queries[
          logging_record.query_record.provider_model] = logging_record
      update_models.provider_queries[
          logging_record.query_record.provider_model] = logging_record
    if self.model_cache_manager:
      self.model_cache_manager.update(
          model_status_updates=update_models, call_type=call_type)
      self.latest_model_cache_path_used_for_update = (
          self.model_cache_manager.cache_path)
    self._update_provider_queries(models)

  def _format_set(
      self,
      provider_model_set: Set[types.ProviderModelType]
  ) -> List[types.ProviderModelType]:
    return sorted(list(provider_model_set))

  def _update_provider_queries(
      self,
      models: types.ModelStatus):
    models.provider_queries = {
        provider_model: query
        for provider_model, query in models.provider_queries.items()
        if provider_model in models.working_models or
        provider_model in models.failed_models
    }

  def _fetch_all_models(
      self,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> types.ModelStatus:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    if self.model_cache_manager and clear_model_cache:
      self.model_cache_manager.clear_cache()

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    models = types.ModelStatus()
    self._load_provider_keys()
    self._get_all_models(models, call_type=call_type)
    self._filter_by_provider_key(models)
    self._filter_by_cache(models, call_type=types.CallType.GENERATE_TEXT)

    print_flag = bool(verbose and models.unprocessed_models)
    if print_flag:
      print(f'From cache;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      print(f'Running test for {len(models.unprocessed_models)} models.')
    self._test_models(
        models,
        call_type=types.CallType.GENERATE_TEXT,
        verbose=verbose)
    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    if print_flag:
      print(f'After test;\n'
            f'  {len(models.working_models)} models are working.\n'
            f'  {len(models.failed_models)} models are failed.')
      duration = (end_utc_date - start_utc_date).total_seconds()
      print(f'Test duration: {duration} seconds.')

    self.has_fetched_all_models = True
    return models

  def _check_model_cache_path_same(self):
    model_cache_path = None
    if not self.model_cache_manager:
      model_cache_path = None
    else:
      model_cache_path = self.model_cache_manager.cache_path
    return model_cache_path == self.latest_model_cache_path_used_for_update

  def list_models(
      self,
      only_largest_models: bool = False,
      verbose: bool = False,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> Union[List[types.ProviderModelType], types.ModelStatus]:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    models: Optional[types.ModelStatus] = None
    if not self.model_cache_manager:
      logging_utils.log_message(
          logging_options=self.logging_options,
          message='Model cache is not enabled. Fetching all models from '
          'providers. This is not ideal for performance.',
          type=types.LoggingType.WARNING)
      models = self._fetch_all_models(
          call_type=call_type,
          verbose=verbose)
    elif (
        clear_model_cache or
        not self._check_model_cache_path_same() or
        not self.has_fetched_all_models):
      models = self._fetch_all_models(
          clear_model_cache=clear_model_cache,
          call_type=call_type,
          verbose=verbose)
    else:
      models = self.model_cache_manager.get(call_type=call_type)

    if only_largest_models:
      models = copy.deepcopy(models)
      self._filter_largest_models(models)

    if return_all:
      return models
    return self._format_set(models.working_models)

  def list_providers(
      self,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> List[str]:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    providers_with_key: Optional[Set[str]] = None
    if not self.model_cache_manager:
      # For performance, we only load the provider keys if the model cache is
      # not enabled instead of fetching all models.
      self._load_provider_keys()
      providers_with_key = self.providers_with_key
    elif (
        clear_model_cache or
        not self._check_model_cache_path_same() or
        not self.has_fetched_all_models):
      models = self._fetch_all_models(
          verbose=verbose,
          clear_model_cache=clear_model_cache,
          call_type=call_type)
      providers_with_key = set([
          model.provider
          for model in models.working_models])
    else:
      models = self.model_cache_manager.get(call_type=call_type)
      providers_with_key = set([
          model.provider
          for model in models.working_models])

    return sorted(list(providers_with_key))

  def list_provider_models(
      self,
      provider: str,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> List[types.ProviderModelType]:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    if provider not in model_configs.GENERATE_TEXT_MODELS:
      raise ValueError(
          f'Provider not found in model_configs: {provider}\n'
          f'Available providers: {model_configs.GENERATE_TEXT_MODELS.keys()}')

    provider_models: Optional[List[types.ProviderModelType]] = None
    if not self.model_cache_manager:
      # For performance, we only load the provider keys if the model cache is
      # not enabled instead of fetching all models.
      self._load_provider_keys()
      if provider not in self.providers_with_key:
        raise ValueError(
            f'Provider key not found in environment variables for {provider}.\n'
            f'Required keys: {model_configs.PROVIDER_KEY_MAP[provider]}')
      provider_models = list(
          model_configs.GENERATE_TEXT_MODELS[provider].values())
    elif (
        clear_model_cache or
        not self._check_model_cache_path_same() or
        not self.has_fetched_all_models):
      models = self._fetch_all_models(
          verbose=verbose,
          clear_model_cache=clear_model_cache,
          call_type=call_type)
      provider_models = []
      for model in models.working_models:
        if model.provider == provider:
          provider_models.append(model)
    else:
      models = self.model_cache_manager.get(call_type=call_type)
      provider_models = []
      for model in models.working_models:
        if model.provider == provider:
          provider_models.append(model)

    return sorted(provider_models)

  def get_model(
      self,
      provider: str,
      model: str,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> types.ProviderModelType:
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    if provider not in model_configs.GENERATE_TEXT_MODELS:
      raise ValueError(
          f'Provider not found in model_configs: {provider}\n'
          f'Available providers: {model_configs.GENERATE_TEXT_MODELS.keys()}')

    if model not in model_configs.GENERATE_TEXT_MODELS[provider]:
      raise ValueError(
          f'Model not found in {provider} models: {model}\n'
          f'Available models: {model_configs.GENERATE_TEXT_MODELS[provider]}')

    if not self.model_cache_manager:
      # For performance, we only load the provider keys if the model cache is
      # not enabled instead of fetching all models.
      self._load_provider_keys()
      if provider not in self.providers_with_key:
        raise ValueError(
            f'Provider key not found in environment variables for {provider}.\n'
            f'Required keys: {model_configs.PROVIDER_KEY_MAP[provider]}')
      return model_configs.GENERATE_TEXT_MODELS[provider][model]
    elif (
        clear_model_cache or
        not self._check_model_cache_path_same() or
        not self.has_fetched_all_models):
      models = self._fetch_all_models(
          verbose=verbose,
          clear_model_cache=clear_model_cache,
          call_type=call_type)
    else:
      models = self.model_cache_manager.get(call_type=call_type)

    for working_model in models.working_models:
      if working_model.provider == provider and working_model.model == model:
        return working_model
    raise ValueError(
        f'Provider model not found in working models: ({provider}, {model})\n'
        f'Available models: {models.working_models}')

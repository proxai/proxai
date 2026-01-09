import dataclasses
import os
import tempfile
from typing import Optional, Union
import platformdirs
import proxai.types as types
import proxai.type_utils as type_utils
import proxai.caching.query_cache as query_cache
import proxai.caching.model_cache as model_cache
import proxai.connections.available_models as available_models
import proxai.connections.proxdash as proxdash
import proxai.experiment.experiment as experiment
import proxai.connectors.model_configs as model_configs
import proxai.state_controllers.state_controller as state_controller
import proxai.serializers.type_serializer as type_serializer


_PROXAI_CLIENT_STATE_PROPERTY = '_proxai_client_state'


@dataclasses.dataclass
class ProxAIClientParams:
    experiment_path: Optional[str] = None
    cache_options: Optional[types.CacheOptions] = None
    logging_options: Optional[types.LoggingOptions] = None
    proxdash_options: Optional[types.ProxDashOptions] = None
    allow_multiprocessing: Optional[bool] = True
    model_test_timeout: Optional[int] = 25
    feature_mapping_strategy: Optional[types.FeatureMappingStrategy] = types.FeatureMappingStrategy.BEST_EFFORT
    suppress_provider_errors: Optional[bool] = False


class ProxAIClient(state_controller.StateControlled):
  def __init__(
      self,
      init_from_params: Optional[ProxAIClientParams] = None,
      init_from_state: Optional[types.ProxAIClientState] = None
  ):
    super().__init__(
        init_from_params=init_from_params,
        init_from_state=init_from_state)

    if init_from_state is not None:
      self.load_state(init_from_state)
    else:
      self._set_default_values()

      self.experiment_path = init_from_params.experiment_path
      self.cache_options = init_from_params.cache_options
      self.logging_options = init_from_params.logging_options
      self.proxdash_options = init_from_params.proxdash_options
      self.allow_multiprocessing = init_from_params.allow_multiprocessing
      self.model_test_timeout = init_from_params.model_test_timeout
      self.feature_mapping_strategy = init_from_params.feature_mapping_strategy
      self.suppress_provider_errors = init_from_params.suppress_provider_errors

      _ = self.model_cache_manager
      _ = self.query_cache_manager
      self._init_proxdash_connection()

      if (self.cache_options and
          self.cache_options.clear_model_cache_on_connect):
        self.model_cache_manager.clear_cache()
      if (self.cache_options and
          self.cache_options.clear_query_cache_on_connect):
        self.query_cache_manager.clear_cache()

      available_models_params = available_models.AvailableModelsParams(
          run_type=self.run_type,
          feature_mapping_strategy=self.feature_mapping_strategy,
          model_configs_instance=self.model_configs_instance,
          model_cache_manager=self.model_cache_manager,
          query_cache_manager=self.query_cache_manager,
          logging_options=self.logging_options,
          proxdash_connection=self.proxdash_connection,
          allow_multiprocessing=self.allow_multiprocessing,
          model_test_timeout=self.model_test_timeout,
      )
      self._available_models_instance = available_models.AvailableModels(
          init_from_params=available_models_params)

  def get_internal_state_property_name(self):
    return _PROXAI_CLIENT_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.ProxAIClientState

  def handle_changes(
      self,
      old_state: types.ProxAIClientState,
      current_state: types.ProxAIClientState
  ):
    pass

  def _init_default_model_cache_manager(self):
    try:
      app_dirs = platformdirs.PlatformDirs(appname="proxai", appauthor="proxai")
      self.default_model_cache_path =  app_dirs.user_cache_dir
      os.makedirs(self.default_model_cache_path, exist_ok=True)
      # 4 hours cache duration makes sense for local development if proxai is
      # using platform app cache directory
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=types.CacheOptions(
              cache_path=self.default_model_cache_path,
              model_cache_duration=60 * 60 * 4))
      self.default_model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params)
      self.platform_used_for_default_model_cache = True
    except Exception as e:
      self.default_model_cache_path = tempfile.TemporaryDirectory()
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=types.CacheOptions(
              cache_path=self.default_model_cache_path.name))
      self.default_model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params)
      self.platform_used_for_default_model_cache = False

  def _set_default_values(self):
    self.run_type = types.RunType.PRODUCTION
    self.hidden_run_key = experiment.get_hidden_run_key()
    self.experiment_path = None
    self.root_logging_path = None

    self.logging_options = None
    self.cache_options = None
    self.proxdash_options = None

    self.model_configs_instance = model_configs.ModelConfigs()
    self.model_configs_requested_from_proxdash = False

    self.registered_model_connectors = {}
    self.model_connectors = {}
    self.model_cache_manager = None
    self.query_cache_manager = None
    self.proxdash_connection = None
    self._init_default_model_cache_manager()

    self.feature_mapping_strategy = types.FeatureMappingStrategy.BEST_EFFORT
    self.suppress_provider_errors = False
    self.allow_multiprocessing = True
    self.model_test_timeout = 25

    self.available_models_instance = None

  @property
  def run_type(self) -> types.RunType:
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, value: types.RunType):
    self.set_property_value('run_type', value)

  @property
  def hidden_run_key(self) -> str:
    return self.get_property_value('hidden_run_key')

  @hidden_run_key.setter
  def hidden_run_key(self, value: str):
    self.set_property_value('hidden_run_key', value)

  @property
  def experiment_path(self) -> Optional[str]:
    return self.get_property_value('experiment_path')

  @experiment_path.setter
  def experiment_path(self, value: Optional[str]):
    if value is not None:
      experiment.validate_experiment_path(value)
    self.set_property_value('experiment_path', value)

  @property
  def default_model_cache_path(self) -> Optional[str]:
    return self.get_property_value('default_model_cache_path')

  @default_model_cache_path.setter
  def default_model_cache_path(self, value: Optional[str]):
    self.set_property_value('default_model_cache_path', value)

  @property
  def root_logging_path(self) -> Optional[str]:
    return self.get_property_value('root_logging_path')

  @root_logging_path.setter
  def root_logging_path(self, value: Optional[str]):
    if value and not os.path.exists(value):
      raise ValueError(
          f'Root logging path does not exist: {value}')
    self.set_property_value('root_logging_path', value)

  @property
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, value: Optional[types.LoggingOptions]):
    root_logging_path = None
    if value and value.logging_path:
      root_logging_path = value.logging_path
    else:
      root_logging_path = None

    result_logging_options = types.LoggingOptions()
    if root_logging_path is not None:
      if self.experiment_path is not None:
        result_logging_options.logging_path = os.path.join(
            root_logging_path, self.experiment_path)
      else:
        result_logging_options.logging_path = root_logging_path
      if not os.path.exists(result_logging_options.logging_path):
        os.makedirs(result_logging_options.logging_path, exist_ok=True)
    else:
      result_logging_options.logging_path = None

    if value is not None:
      result_logging_options.stdout = value.stdout
      result_logging_options.hide_sensitive_content = (
          value.hide_sensitive_content)

    self.set_property_value('logging_options', result_logging_options)

  @property
  def cache_options(self) -> types.CacheOptions:
    return self.get_property_value('cache_options')

  @cache_options.setter
  def cache_options(self, value: Optional[types.CacheOptions]):
    result_cache_options = None
    if value is not None:
      if not value.cache_path:
        raise ValueError('cache_path is required while setting cache_options')
      result_cache_options = types.CacheOptions()

      result_cache_options.cache_path = value.cache_path

      result_cache_options.unique_response_limit = (
          value.unique_response_limit)
      result_cache_options.retry_if_error_cached = (
          value.retry_if_error_cached)
      result_cache_options.clear_query_cache_on_connect = (
          value.clear_query_cache_on_connect)

      result_cache_options.disable_model_cache = (
          value.disable_model_cache)
      result_cache_options.clear_model_cache_on_connect = (
          value.clear_model_cache_on_connect)
      result_cache_options.model_cache_duration = (
          value.model_cache_duration)

    self.set_property_value('cache_options', result_cache_options)

  @property
  def proxdash_options(self) -> types.ProxDashOptions:
    return self.get_property_value('proxdash_options')

  @proxdash_options.setter
  def proxdash_options(self, value: Optional[types.ProxDashOptions]):
    result_proxdash_options = types.ProxDashOptions()
    if value is not None:
      result_proxdash_options.stdout = value.stdout
      result_proxdash_options.hide_sensitive_content = (
          value.hide_sensitive_content)
      result_proxdash_options.disable_proxdash = value.disable_proxdash
      result_proxdash_options.api_key = value.api_key
      result_proxdash_options.base_url = value.base_url

    self.set_property_value('proxdash_options', result_proxdash_options)

  @property
  def model_configs_requested_from_proxdash(self) -> bool:
    return self.get_property_value('model_configs_requested_from_proxdash')

  @model_configs_requested_from_proxdash.setter
  def model_configs_requested_from_proxdash(self, value: bool):
    self.set_property_value('model_configs_requested_from_proxdash', value)

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
    if value < 1:
      raise ValueError('model_test_timeout must be greater than 0.')
    self.set_property_value('model_test_timeout', value)

  @property
  def feature_mapping_strategy(self) -> types.FeatureMappingStrategy:
    return self.get_property_value('feature_mapping_strategy')

  @feature_mapping_strategy.setter
  def feature_mapping_strategy(self, value: types.FeatureMappingStrategy):
    self.set_property_value('feature_mapping_strategy', value)

  @property
  def suppress_provider_errors(self) -> bool:
    return self.get_property_value('suppress_provider_errors')

  @suppress_provider_errors.setter
  def suppress_provider_errors(self, value: bool):
    self.set_property_value('suppress_provider_errors', value)

  @property
  def model_configs_instance(self) -> model_configs.ModelConfigs:
    if (not self.model_configs_requested_from_proxdash and
        self.proxdash_connection):
      model_configs_schema = self.proxdash_connection.get_model_configs_schema()
      if model_configs_schema is not None:
        self._model_configs_instance.model_configs_schema = model_configs_schema
      self.model_configs_requested_from_proxdash = True
    return self.get_property_value('model_configs_instance')

  @model_configs_instance.setter
  def model_configs_instance(self, value: model_configs.ModelConfigs):
    self.set_property_value('model_configs_instance', value)

  @property
  def default_model_cache_manager(self) -> model_cache.ModelCacheManager:
    return self.get_property_value('default_model_cache_manager')

  @default_model_cache_manager.setter
  def default_model_cache_manager(self, value: model_cache.ModelCacheManager):
    self.set_property_value('default_model_cache_manager', value)

  @property
  def model_cache_manager(self) -> model_cache.ModelCacheManager:
    if self.cache_options is None:
      return self.default_model_cache_manager

    if self._model_cache_manager is None:
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=self.cache_options)
      self._model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params)
    if (self._model_cache_manager.status !=
        types.ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND):
      return self._model_cache_manager

    return None

  @model_cache_manager.setter
  def model_cache_manager(self, value: model_cache.ModelCacheManager):
    self.set_property_value('model_cache_manager', value)

  @property
  def query_cache_manager(self) -> query_cache.QueryCacheManager:
    if (self._query_cache_manager is None and
        self.cache_options is not None):
      query_cache_manager_params = query_cache.QueryCacheManagerParams(
          cache_options=self.cache_options)
      self._query_cache_manager = query_cache.QueryCacheManager(
          init_from_params=query_cache_manager_params)
    return self.get_property_value('query_cache_manager')

  @query_cache_manager.setter
  def query_cache_manager(self, value: query_cache.QueryCacheManager):
    self.set_property_value('query_cache_manager', value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value: proxdash.ProxDashConnection):
    self.set_property_value('proxdash_connection', value)

  @property
  def available_models_instance(self) -> available_models.AvailableModels:
    return self.get_state_controlled_property_value('available_models_instance')

  @available_models_instance.setter
  def available_models_instance(self, value: available_models.AvailableModels):
    self.set_state_controlled_property_value('available_models_instance', value)

  def _init_proxdash_connection(self):
    if self._proxdash_connection is None:
      proxdash_connection_params = proxdash.ProxDashConnectionParams(
          hidden_run_key=self.hidden_run_key,
          experiment_path=self.experiment_path,
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options)
      self._proxdash_connection = proxdash.ProxDashConnection(
          init_from_params=proxdash_connection_params)

  def get_registered_model_connector(
      self,
      call_type: types.CallType
  ):
    if call_type != types.CallType.GENERATE_TEXT:
      raise ValueError(f'Call type not supported: {call_type}')

    if call_type not in self.registered_model_connectors:
      for provider_model in self.model_configs_instance.get_default_model_priority_list():
        try:
          self.available_models_instance.get_working_model(
              provider=provider_model.provider,
              model=provider_model.model)
          self.registered_model_connectors[
              call_type] = self.available_models_instance.get_model_connector(
                  provider_model)
          break
        except ValueError:
          continue

      if call_type not in self.registered_model_connectors:
        models = self.available_models_instance.list_working_models()
        if len(models.working_models) > 0:
          self.registered_model_connectors[
              call_type] = self.available_models_instance.get_model_connector(
                  models.working_models.pop())
        else:
          raise ValueError(
              'No working models found in current environment:\n'
              '* Please check your environment variables and try again.\n'
              '* You can use px.check_health() method as instructed in '
              'https://www.proxai.co/proxai-docs/check-health')

    return self.registered_model_connectors[call_type]

  def set_model(
      self,
      provider_model: Optional[types.ProviderModelIdentifierType] = None,
      generate_text: Optional[types.ProviderModelIdentifierType] = None):
    if provider_model and generate_text:
      raise ValueError('provider_model and generate_text cannot be set at the '
                      'same time. Please set one of them.')

    if provider_model is None and generate_text is None:
      raise ValueError('provider_model or generate_text must be set.')

    if generate_text:
      provider_model = generate_text

    self.model_configs_instance.check_provider_model_identifier_type(provider_model)

    self.registered_model_connectors[
        types.CallType.GENERATE_TEXT
    ] = self.available_models_instance.get_model_connector(provider_model)

  def generate_text(
      self,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = None,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None,
      response_format: Optional[types.ResponseFormatParam] = None,
      web_search: Optional[bool] = None,
      provider_model: Optional[types.ProviderModelIdentifierType] = None,
      feature_mapping_strategy: Optional[types.FeatureMappingStrategy] = None,
      use_cache: Optional[bool] = None,
      unique_response_limit: Optional[int] = None,
      extensive_return: bool = False,
      suppress_provider_errors: Optional[bool] = None) -> Union[str, types.LoggingRecord]:
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages is not None:
      type_utils.check_messages_type(messages)

    if use_cache:
      if self.query_cache_manager is None:
        raise ValueError(
            'use_cache is True but query cache is not working.\n'
            'Please set query cache options to enable query cache.')
      if (self.query_cache_manager.status !=
          types.QueryCacheManagerStatus.WORKING):
        raise ValueError(
            'use_cache is True but query cache is not working.\n'
            f'Query Cache Manager Status: {self.query_cache_manager.status}')
    elif use_cache is None:
      use_cache = (
          self.query_cache_manager is not None and
          self.query_cache_manager.status ==
          types.QueryCacheManagerStatus.WORKING)

    if provider_model is not None:
      model_connector = self.available_models_instance.get_model_connector(
          provider_model_identifier=provider_model)
    else:
      model_connector = self.get_registered_model_connector(
          call_type=types.CallType.GENERATE_TEXT)

    if suppress_provider_errors is None:
      suppress_provider_errors = self.suppress_provider_errors

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
        web_search=web_search,
        feature_mapping_strategy=feature_mapping_strategy,
        use_cache=use_cache,
        unique_response_limit=unique_response_limit)
    if (logging_record.response_record.error or
        logging_record.response_record.error_traceback):
      if suppress_provider_errors:
        if extensive_return:
          return logging_record
        return logging_record.response_record.error
      else:
        error_traceback = ''
        if logging_record.response_record.error_traceback:
          error_traceback = logging_record.response_record.error_traceback + '\n'
        raise Exception(error_traceback + logging_record.response_record.error)

    if (logging_record.response_record.response.type ==
        types.ResponseType.PYDANTIC):
      # Recreate instance from pydantic_metadata if value is None (from cache)
      instance = type_utils.create_pydantic_instance_from_response(
          response_format=response_format,
          response=logging_record.response_record.response)
      logging_record.response_record.response.value = instance

    if extensive_return:
      return logging_record

    return logging_record.response_record.response.value

  def get_current_options(
      self,
      json: bool = False):
    run_options = types.RunOptions(
        run_type=self.run_type,
        hidden_run_key=self.hidden_run_key,
        experiment_path=self.experiment_path,
        root_logging_path=self.root_logging_path,
        default_model_cache_path=self.default_model_cache_path,
        logging_options=self.logging_options,
        cache_options=self.cache_options,
        proxdash_options=self.proxdash_options,
        feature_mapping_strategy=self.feature_mapping_strategy,
        suppress_provider_errors=self.suppress_provider_errors,
        allow_multiprocessing=self.allow_multiprocessing,
        model_test_timeout=self.model_test_timeout)
    if json:
      return type_serializer.encode_run_options(run_options=run_options)
    return run_options

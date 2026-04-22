from __future__ import annotations

import copy
import dataclasses
import datetime
import multiprocessing
import os
import traceback
from collections.abc import Callable

import proxai.caching.model_cache as model_cache
import proxai.caching.query_cache as query_cache
import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.proxdash as proxdash
import proxai.connectors.files as files_module
import proxai.connectors.model_configs as model_configs
import proxai.connectors.model_registry as model_registry
import proxai.connectors.provider_connector as provider_connector
import proxai.logging.utils as logging_utils
import proxai.state_controllers.state_controller as state_controller
import proxai.connectors.adapter_utils as adapter_utils
import proxai.type_utils as type_utils
import proxai.types as types

_AVAILABLE_MODELS_STATE_PROPERTY = '_available_models_state'
_GENERATE_TEXT_TEST_PROMPT = 'Hello model!'
_GENERATE_TEXT_TEST_MAX_TOKENS = 1000


@dataclasses.dataclass
class AvailableModelsParams:
  """Initialization parameters for AvailableModels."""

  run_type: types.RunType | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  model_configs_instance: model_configs.ModelConfigs | None = None
  model_cache_manager: model_cache.ModelCacheManager | None = None
  query_cache_manager: query_cache.QueryCacheManager | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  api_key_manager: api_key_manager.ApiKeyManager | None = None
  files_manager: files_module.FilesManager | None = None
  model_probe_options: types.ModelProbeOptions | None = None
  debug_options: types.DebugOptions | None = None


class AvailableModels(state_controller.StateControlled):
  """Discovers and manages available AI models across providers."""

  _model_cache_manager: model_cache.ModelCacheManager | None
  _run_type: types.RunType
  _model_configs_instance: model_configs.ModelConfigs
  _cache_options: types.CacheOptions
  _logging_options: types.LoggingOptions
  _provider_call_options: types.ProviderCallOptions
  _model_probe_options: types.ModelProbeOptions
  _debug_options: types.DebugOptions
  _proxdash_connection: proxdash.ProxDashConnection
  _api_key_manager: api_key_manager.ApiKeyManager | None
  _latest_model_cache_path_used_for_update: str | None
  _available_models_state: types.AvailableModelsState
  provider_connectors: dict[str, provider_connector.ProviderConnector]

  def __init__(
      self, init_from_params: AvailableModelsParams | None = None,
      init_from_state: types.AvailableModelsState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    self.provider_connectors = {}

    if init_from_state:
      self.load_state(init_from_state)
    else:
      self.run_type = init_from_params.run_type
      self.provider_call_options = init_from_params.provider_call_options
      self.model_configs_instance = init_from_params.model_configs_instance

      self.model_cache_manager = init_from_params.model_cache_manager
      self.query_cache_manager = init_from_params.query_cache_manager
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.api_key_manager = init_from_params.api_key_manager
      self.files_manager_instance = init_from_params.files_manager
      self.model_probe_options = init_from_params.model_probe_options
      self.debug_options = init_from_params.debug_options

      self.latest_model_cache_path_used_for_update = None

  def get_internal_state_property_name(self) -> str:
    """Return the name of the internal state property."""
    return _AVAILABLE_MODELS_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
    """Return the dataclass type used for state storage."""
    return types.AvailableModelsState

  def get_model_connector(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType,
  ) -> provider_connector.ProviderConnector:
    """Get or create a provider-scoped connector for the specified model."""
    provider_model = self.model_configs_instance.get_provider_model(
        provider_model_identifier
    )
    provider = provider_model.provider
    if provider in self.provider_connectors:
      return self.provider_connectors[provider]

    connector = model_registry.get_model_connector(provider=provider)

    init_from_params = connector.keywords['init_from_params']
    init_from_params.run_type = self.run_type
    init_from_params.provider_call_options = self.provider_call_options
    init_from_params.query_cache_manager = self.query_cache_manager
    init_from_params.logging_options = self.logging_options
    init_from_params.proxdash_connection = self.proxdash_connection
    init_from_params.provider_token_value_map = (
        self.api_key_manager.get_provider_keys(provider)
    )
    init_from_params.debug_options = self.debug_options
    init_from_params.files_manager = self.files_manager_instance

    self.provider_connectors[provider] = connector()
    return self.provider_connectors[provider]

  @property
  def run_type(self) -> types.RunType:
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, run_type: types.RunType):
    self.set_property_value('run_type', run_type)

  @property
  def model_configs_instance(self) -> model_configs.ModelConfigs:
    return self.get_state_controlled_property_value('model_configs_instance')

  @model_configs_instance.setter
  def model_configs_instance(
      self, model_configs_instance: model_configs.ModelConfigs
  ):
    self.set_state_controlled_property_value(
        'model_configs_instance', model_configs_instance
    )

  def model_configs_instance_deserializer(
      self, state_value: types.ModelConfigsState
  ) -> model_configs.ModelConfigs:
    return model_configs.ModelConfigs(init_from_state=state_value)

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
  def model_cache_manager(self, value: model_cache.ModelCacheManager):
    self.set_state_controlled_property_value('model_cache_manager', value)

  def model_cache_manager_deserializer(
      self, state_value: types.ModelCacheManagerState
  ) -> model_cache.ModelCacheManager:
    return model_cache.ModelCacheManager(init_from_state=state_value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value: proxdash.ProxDashConnection):
    self.set_state_controlled_property_value('proxdash_connection', value)

  def proxdash_connection_deserializer(
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def api_key_manager(self) -> api_key_manager.ApiKeyManager:
    return self.get_state_controlled_property_value('api_key_manager')

  @api_key_manager.setter
  def api_key_manager(self, value: api_key_manager.ApiKeyManager):
    self.set_state_controlled_property_value('api_key_manager', value)

  def api_key_manager_deserializer(
      self, state_value: types.ApiKeyManagerState
  ) -> api_key_manager.ApiKeyManager:
    return api_key_manager.ApiKeyManager(init_from_state=state_value)

  @property
  def files_manager_instance(self) -> files_module.FilesManager:
    return self.get_state_controlled_property_value(
        'files_manager_instance')

  @files_manager_instance.setter
  def files_manager_instance(self, value: files_module.FilesManager):
    self.set_state_controlled_property_value(
        'files_manager_instance', value)

  def files_manager_instance_deserializer(
      self, state_value: types.FilesManagerState
  ) -> files_module.FilesManager:
    return files_module.FilesManager(init_from_state=state_value)

  @property
  def provider_call_options(self) -> types.ProviderCallOptions:
    return self.get_property_value('provider_call_options')

  @provider_call_options.setter
  def provider_call_options(self, value: types.ProviderCallOptions):
    self.set_property_value('provider_call_options', value)

  @property
  def model_probe_options(self) -> types.ModelProbeOptions:
    return self.get_property_value('model_probe_options')

  @model_probe_options.setter
  def model_probe_options(self, value: types.ModelProbeOptions):
    self.set_property_value('model_probe_options', value)

  @property
  def query_cache_manager(self) -> query_cache.QueryCacheManager:
    return self.get_state_controlled_property_value('query_cache_manager')

  @query_cache_manager.setter
  def query_cache_manager(self, value: query_cache.QueryCacheManager):
    self.set_state_controlled_property_value('query_cache_manager', value)

  def query_cache_manager_deserializer(
      self, state_value: types.QueryCacheManagerState
  ) -> query_cache.QueryCacheManager:
    return query_cache.QueryCacheManager(init_from_state=state_value)

  @property
  def debug_options(self) -> types.DebugOptions:
    return self.get_property_value('debug_options')

  @debug_options.setter
  def debug_options(self, value: types.DebugOptions):
    self.set_property_value('debug_options', value)

  @property
  def latest_model_cache_path_used_for_update(self) -> str | None:
    return self.get_property_value('latest_model_cache_path_used_for_update')

  @latest_model_cache_path_used_for_update.setter
  def latest_model_cache_path_used_for_update(self, value: str | None):
    self.set_property_value('latest_model_cache_path_used_for_update', value)

  def _get_all_models(
      self, models: types.ModelStatus, recommended_only: bool = True
  ):
    provider_models = self.model_configs_instance.get_all_models(
        recommended_only=recommended_only
    )
    for provider_model in provider_models:
      models.unprocessed_models.add(provider_model)

  def _filter_by_provider_api_key(self, models: types.ModelStatus):

    def _filter_set(
        provider_model_set: set[types.ProviderModelType]
    ) -> tuple[set[types.ProviderModelType], set[types.ProviderModelType]]:
      not_filtered_models = set()
      for provider_model in provider_model_set:
        if provider_model.provider not in self.api_key_manager.providers_with_key:
          models.filtered_models.add(provider_model)
        else:
          not_filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_providers(
      self, models: types.ModelStatus, providers: set[str] | None = None
  ):
    if providers is None:
      return

    def _filter_set(
        provider_model_set: set[types.ProviderModelType]
    ) -> tuple[set[types.ProviderModelType], set[types.ProviderModelType]]:
      not_filtered_models = set()
      for provider_model in provider_model_set:
        if provider_model.provider not in providers:
          models.filtered_models.add(provider_model)
        else:
          not_filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_provider_models(
      self, models: types.ModelStatus,
      provider_models: set[types.ProviderModelType] | None = None
  ):
    if provider_models is None:
      return

    def _filter_set(
        provider_model_set: set[types.ProviderModelType]
    ) -> tuple[set[types.ProviderModelType], set[types.ProviderModelType]]:
      not_filtered_models = set()
      for provider_model in provider_model_set:
        if provider_model not in provider_models:
          models.filtered_models.add(provider_model)
        else:
          not_filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_model_size(
      self, models: types.ModelStatus,
      model_size: types.ModelSizeType | None = None,
      recommended_only: bool = True
  ):
    if model_size is None:
      return

    allowed_models = self.model_configs_instance.get_all_models(
        model_size=model_size, recommended_only=recommended_only
    )

    def _filter_set(
        provider_model_set: set[types.ProviderModelType]
    ) -> tuple[set[types.ProviderModelType], set[types.ProviderModelType]]:
      not_filtered_models = set()
      for provider_model in provider_model_set:
        if provider_model in allowed_models:
          not_filtered_models.add(provider_model)
        else:
          models.filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _is_feature_compatible(
      self, support_level: types.FeatureSupportType
  ) -> bool:
    if support_level == types.FeatureSupportType.SUPPORTED:
      return True
    if support_level == types.FeatureSupportType.BEST_EFFORT:
      return (
          self.provider_call_options.feature_mapping_strategy
          == types.FeatureMappingStrategy.BEST_EFFORT or
          self.provider_call_options.feature_mapping_strategy is None
      )
    return False

  def _filter_by_tag_list(
      self,
      models: types.ModelStatus,
      tags: list,
      resolve_fn: Callable,
  ) -> None:
    if tags is None:
      return

    def _filter_set(provider_model_set):
      not_filtered_models = set()
      for provider_model in provider_model_set:
        provider_model_config = (
            self.model_configs_instance.
            get_provider_model_config(provider_model)
        )
        support_level = self.get_model_connector(
            provider_model
        ).get_tag_support_level(
            tags=tags, resolve_fn=resolve_fn,
            model_feature_config=provider_model_config.features
        )
        if self._is_feature_compatible(support_level):
          not_filtered_models.add(provider_model)
        else:
          models.filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_input_format(
      self, models: types.ModelStatus,
      input_format_types: list[types.InputFormatType] | None = None
  ):
    if input_format_types is None:
      return
    self._filter_by_tag_list(
        models, input_format_types,
        adapter_utils.resolve_input_format_type_support
    )

  def _filter_by_output_format(
      self, models: types.ModelStatus,
      output_format_types: list[types.OutputFormatType] | None = None
  ):
    if output_format_types is None:
      return
    self._filter_by_tag_list(
        models, output_format_types,
        adapter_utils.resolve_output_format_type_support
    )

  def _filter_by_feature_tags(
      self, models: types.ModelStatus,
      feature_tags: list[types.FeatureTag] | None = None
  ):
    if feature_tags is None:
      return

    def _filter_set(
        provider_model_set: set[types.ProviderModelType]
    ) -> set[types.ProviderModelType]:
      not_filtered_models = set()
      for provider_model in provider_model_set:
        provider_model_config = (
            self.model_configs_instance.
            get_provider_model_config(provider_model)
        )
        support_level = self.get_model_connector(
            provider_model
        ).get_feature_tags_support_level(
            feature_tags=feature_tags,
            model_feature_config=provider_model_config.features
        )
        if self._is_feature_compatible(support_level):
          not_filtered_models.add(provider_model)
        else:
          models.filtered_models.add(provider_model)
      return not_filtered_models

    models.unprocessed_models = _filter_set(models.unprocessed_models)
    models.working_models = _filter_set(models.working_models)
    models.failed_models = _filter_set(models.failed_models)

  def _filter_by_tool_tags(
      self, models: types.ModelStatus,
      tool_tags: list[types.ToolTag] | None = None
  ):
    if tool_tags is None:
      return
    self._filter_by_tag_list(
        models, tool_tags, adapter_utils.resolve_tool_tag_support
    )

  def _filter_by_cache(
      self, models: types.ModelStatus,
      output_format_type: types.OutputFormatType
  ):
    if (
        not self.model_cache_manager or
        self.model_cache_manager.status != types.ModelCacheManagerStatus.WORKING
    ):
      return
    cache_model_status = types.ModelStatus()
    if self.model_cache_manager:
      cache_model_status = self.model_cache_manager.get(
          output_format_type=output_format_type
      )
      self._place_filtered_models_back_to_working_or_failed_models(
          cache_model_status
      )

    for provider_model, query in cache_model_status.provider_queries.items():
      models.provider_queries[provider_model] = copy.deepcopy(query)

    for provider_model in list(models.unprocessed_models):
      if provider_model in cache_model_status.working_models:
        models.unprocessed_models.discard(provider_model)
        models.working_models.add(provider_model)
      elif provider_model in cache_model_status.failed_models:
        models.unprocessed_models.discard(provider_model)
        models.failed_models.add(provider_model)

  @staticmethod
  def _test_generate_text(
      provider_state: types.ProviderState,
      provider_model: types.ProviderModelType, verbose: bool = False,
      model_configs_state: types.ModelConfigsState | None = None,
      model_configs_instance: model_configs.ModelConfigs | None = None
  ) -> types.CallRecord:
    if verbose:
      print(f'Testing {provider_model}...')
    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    if model_configs_instance is None:
      model_configs_instance = model_configs.ModelConfigs(
          init_from_state=model_configs_state
      )
    provider_model_config = model_configs_instance.get_provider_model_config(
        provider_model
    )
    connector = model_registry.get_model_connector(
        provider=provider_model.provider, without_additional_args=True
    )
    connector = connector(init_from_state=provider_state)
    try:
      call_record: types.CallRecord = connector.generate(
          prompt=_GENERATE_TEXT_TEST_PROMPT,
          provider_model=provider_model,
          provider_model_config=provider_model_config,
          parameters=types.ParameterType(
              max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS
          ),
          connection_options=types.ConnectionOptions(
              skip_cache=True, suppress_provider_errors=True
          ),
      )
      return call_record
    except Exception as e:
      end_utc_date = datetime.datetime.now(datetime.timezone.utc)
      return types.CallRecord(
          query=types.QueryRecord(
              provider_model=provider_model,
              prompt=_GENERATE_TEXT_TEST_PROMPT,
              parameters=types.ParameterType(
                  max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS
              ),
          ),
          result=types.ResultRecord(
              status=types.ResultStatusType.FAILED,
              error=str(e),
              error_traceback=traceback.format_exc(),
              timestamp=types.TimeStampType(
                  start_utc_date=start_utc_date,
                  end_utc_date=end_utc_date,
                  response_time=(end_utc_date - start_utc_date),
              ),
          ),
          connection=types.ConnectionMetadata(
              result_source=types.ResultSource.PROVIDER
          ),
      )

  def _get_timeout_call_record(self, provider_model: types.ProviderModelType):
    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    start_utc_date = end_utc_date - datetime.timedelta(
        seconds=self.model_probe_options.timeout
    )
    return types.CallRecord(
        query=types.QueryRecord(
            provider_model=provider_model,
            prompt=_GENERATE_TEXT_TEST_PROMPT,
            parameters=types.ParameterType(
                max_tokens=_GENERATE_TEXT_TEST_MAX_TOKENS
            ),
        ),
        result=types.ResultRecord(
            status=types.ResultStatusType.FAILED,
            error=(
                f'Model {provider_model} took longer than '
                f'{self.model_probe_options.timeout} seconds to respond'
            ),
            error_traceback=traceback.format_exc(),
            timestamp=types.TimeStampType(
                start_utc_date=start_utc_date,
                end_utc_date=end_utc_date,
                response_time=(end_utc_date - start_utc_date),
            ),
        ),
        connection=types.ConnectionMetadata(
            result_source=types.ResultSource.PROVIDER
        ),
    )

  def _get_bootstrap_error(self, e: Exception):
    error_str = str(e).lower()
    is_bootstrapping_error = (
        "an attempt has been made to start a new process before" in error_str
        and "current process has finished its bootstrapping phase" in error_str
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
          'model_probe_options=px.ModelProbeOptions('
          'allow_multiprocessing=False) on px.connect()\n'
          'For more details, see followings:\n'
          '- https://www.proxai.co/proxai-docs/advanced/multiprocessing\n'
          '- https://docs.python.org/3/library/multiprocessing.html#the-'
          'spawn-and-forkserver-start-methods\n'
      )
    return None

  def _test_models_with_multiprocessing(
      self, test_tasks: list[tuple[types.ProviderModelType,
                                   provider_connector.ProviderConnector]],
      output_format_type: types.OutputFormatType, verbose: bool = False
  ):
    process_count = max(1, multiprocessing.cpu_count() - 1)
    test_func = None
    if output_format_type == types.OutputFormatType.TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(
          f'Output format type not supported: {output_format_type}'
      )

    test_results = []
    try:
      pool = multiprocessing.Pool(processes=process_count)
      pool_results = []
      model_configs_state = self.model_configs_instance.get_state()
      for provider_model, connector in test_tasks:
        pool_result = pool.apply_async(
            test_func, args=(connector.get_state(), provider_model, verbose),
            kwds={
                'model_configs_state': model_configs_state,
            }
        )
        pool_results.append((provider_model, pool_result))
      pool.close()
      for provider_model, pool_result in pool_results:
        try:
          test_results.append(
              pool_result.get(timeout=self.model_probe_options.timeout)
          )
        except multiprocessing.TimeoutError:
          if verbose:
            print(
                f"> {provider_model} query took longer than "
                f'{self.model_probe_options.timeout} seconds to respond'
            )
          test_results.append(self._get_timeout_call_record(provider_model))
      pool.terminate()
      pool.join()
    except Exception as e:
      bootstrap_error = self._get_bootstrap_error(e)
      if bootstrap_error:
        raise bootstrap_error from e
      else:
        raise
    return test_results

  def _test_models_sequentially(
      self, test_tasks: list[tuple[types.ProviderModelType,
                                   provider_connector.ProviderConnector]],
      output_format_type: types.OutputFormatType, verbose: bool = False
  ):
    """Tests provider models sequentially.

    This function is used when multiprocessing is disabled. Note that this
    function cannot handle model timeouts because of the python's limitation
    on timeout handling without multiprocessing/threading.
    """
    test_func = None
    if output_format_type == types.OutputFormatType.TEXT:
      test_func = self._test_generate_text
    else:
      raise ValueError(
          f'Output format type not supported: {output_format_type}'
      )

    warning_message = (
        'Testing models sequentially can take a while because it '
        'is not possible to handle model timeouts without multiprocessing.\n'
        'Some models may take very long time to respond. '
        'To speed up the test, set model_probe_options='
        'ModelProbeOptions(allow_multiprocessing=True).'
    )
    logging_utils.log_message(
        logging_options=self.logging_options, message=warning_message,
        type=types.LoggingType.WARNING
    )
    # Todo: After adding more stdout control on px.types.LoggingOptions,
    #       following can be removed.
    if verbose:
      print(f'WARNING: {warning_message}')

    test_results = []
    for provider_model, connector in test_tasks:
      test_results.append(
          test_func(
              connector.get_state(), provider_model, verbose,
              model_configs_instance=self.model_configs_instance
          )
      )

    return test_results

  def _test_models(
      self, models: types.ModelStatus,
      output_format_type: types.OutputFormatType, verbose: bool = False
  ):
    if not models.unprocessed_models:
      return

    test_tasks = []
    for provider_model in models.unprocessed_models:
      connector = self.get_model_connector(provider_model)
      test_tasks.append((provider_model, connector))

    if self.model_probe_options.allow_multiprocessing:
      test_results = self._test_models_with_multiprocessing(
          test_tasks=test_tasks, output_format_type=output_format_type,
          verbose=verbose
      )
    else:
      test_results = self._test_models_sequentially(
          test_tasks=test_tasks, output_format_type=output_format_type,
          verbose=verbose
      )

    for call_record in test_results:
      models.unprocessed_models.discard(call_record.query.provider_model)
      if call_record.result.status == types.ResultStatusType.SUCCESS:
        models.working_models.add(call_record.query.provider_model)
      else:
        models.failed_models.add(call_record.query.provider_model)
      models.provider_queries[call_record.query.provider_model] = call_record

  def _format_set(
      self, provider_model_set: set[types.ProviderModelType]
  ) -> list[types.ProviderModelType]:
    return sorted(provider_model_set)

  def _place_filtered_models_back_to_working_or_failed_models(
      self, models: types.ModelStatus
  ):
    for provider_model in list(models.filtered_models):
      if provider_model in models.provider_queries:
        call_record = models.provider_queries[provider_model]
        if call_record.result.status == types.ResultStatusType.SUCCESS:
          models.filtered_models.discard(provider_model)
          models.working_models.add(provider_model)
        elif call_record.result.status == types.ResultStatusType.FAILED:
          models.filtered_models.discard(provider_model)
          models.failed_models.add(provider_model)

  def _fetch_all_models(
      self, selected_providers: set[str] | None = None,
      selected_provider_models: set[types.ProviderModelType] | None = None,
      model_size: types.ModelSizeType | None = None,
      output_format: types.OutputFormatTypeParam | None = None,
      input_format: types.InputFormatTypeParam | None = None,
      feature_tags: list[types.FeatureTag] | None = None,
      tool_tags: list[types.ToolTag] | None = None, verbose: bool = False,
      clear_model_cache: bool = False,
      raw_config_results_without_test: bool = False,
      recommended_only: bool = True
  ) -> types.ModelStatus:
    if self.model_cache_manager and clear_model_cache:
      self.model_cache_manager.clear_cache()

    start_utc_date = datetime.datetime.now(datetime.timezone.utc)
    models = types.ModelStatus()
    self.api_key_manager.load_provider_keys()
    if not self.api_key_manager.providers_with_key:
      raise ValueError(
          'No provider API keys found in environment variables.\n'
          'Please follow the instructions in '
          'https://www.proxai.co/proxai-docs/provider-integrations '
          'to set the environment variables.'
      )
    self._get_all_models(models, recommended_only=recommended_only)
    self._filter_by_provider_api_key(models)
    self._filter_by_providers(models, providers=selected_providers)
    self._filter_by_provider_models(
        models, provider_models=selected_provider_models
    )
    self._filter_by_model_size(
        models, model_size=model_size, recommended_only=recommended_only
    )
    output_format_tags = type_utils.create_output_format_type_list(
        output_format
    )
    input_format_tags = type_utils.create_input_format_type_list(input_format)
    self._filter_by_output_format(models, output_format_tags)
    self._filter_by_input_format(models, input_format_tags)
    self._filter_by_feature_tags(models, feature_tags)
    self._filter_by_tool_tags(models, tool_tags)

    if raw_config_results_without_test:
      return models

    self._filter_by_cache(
        models, output_format_type=types.OutputFormatType.TEXT
    )

    print_flag = bool(verbose and models.unprocessed_models)
    verbose_print = print if print_flag else lambda *args, **kwargs: None

    verbose_print(
        f'From cache;\n'
        f'  {len(models.working_models)} models are working.\n'
        f'  {len(models.failed_models)} models are failed.'
    )

    if models.unprocessed_models:
      verbose_print(
          f'Running test for {len(models.unprocessed_models)} models.'
      )
      self._test_models(
          models, output_format_type=types.OutputFormatType.TEXT,
          verbose=verbose
      )
      verbose_print(
          f'After test;\n'
          f'  {len(models.working_models)} models are working.\n'
          f'  {len(models.failed_models)} models are failed.'
      )

    if self.model_cache_manager:
      self.model_cache_manager.save(
          model_status=models, output_format_type=types.OutputFormatType.TEXT
      )
      self.latest_model_cache_path_used_for_update = (
          self.model_cache_manager.cache_path
      )

    end_utc_date = datetime.datetime.now(datetime.timezone.utc)
    duration = (end_utc_date - start_utc_date).total_seconds()
    verbose_print(f'Test duration: {duration} seconds.')

    return models

  def _check_model_cache_path_same(self):
    model_cache_path = None
    if not self.model_cache_manager:
      model_cache_path = None
    else:
      model_cache_path = self.model_cache_manager.cache_path
    return model_cache_path == self.latest_model_cache_path_used_for_update

  def list_models(
      self, model_size: types.ModelSizeIdentifierType | None = None,
      input_format: types.InputFormatTypeParam | None = None,
      output_format: types.OutputFormatTypeParam = (
          types.OutputFormatType.TEXT
      ), feature_tags: types.FeatureTagParam | None = None,
      tool_tags: types.ToolTagParam | None = None, recommended_only: bool = True
  ) -> list[types.ProviderModelType]:
    """List all configured models matching the filters."""
    if model_size is not None:
      model_size = type_utils.check_model_size_identifier_type(model_size)
    feature_tag_list = None
    if feature_tags is not None:
      feature_tag_list = type_utils.create_feature_tag_list(
          features=feature_tags
      )
    tool_tag_list = type_utils.create_tool_tag_list(tool_tags)

    model_status = self._fetch_all_models(
        model_size=model_size, output_format=output_format,
        input_format=input_format, feature_tags=feature_tag_list,
        tool_tags=tool_tag_list, raw_config_results_without_test=True,
        recommended_only=recommended_only
    )

    return self._format_set(model_status.unprocessed_models)

  def list_providers(
      self, output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True
  ) -> list[str]:
    """List all providers with available API keys."""
    model_status = self._fetch_all_models(
        output_format=output_format, raw_config_results_without_test=True,
        recommended_only=recommended_only
    )
    providers_with_key = {
        model.provider for model in model_status.unprocessed_models
    }

    return sorted(providers_with_key)

  def list_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      input_format: types.InputFormatTypeParam | None = None,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      feature_tags: types.FeatureTagParam | None = None,
      tool_tags: types.ToolTagParam | None = None,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType]:
    """List all models for a specific provider."""
    if model_size is not None:
      model_size = type_utils.check_model_size_identifier_type(model_size)
    feature_tag_list = None
    if feature_tags is not None:
      feature_tag_list = type_utils.create_feature_tag_list(
          features=feature_tags
      )
    tool_tag_list = type_utils.create_tool_tag_list(tool_tags)

    self.api_key_manager.load_provider_keys()
    if provider not in self.api_key_manager.providers_with_key:
      raise ValueError(
          f'Provider key not found in environment variables for {provider}.\n'
          f'Required keys: {model_configs.PROVIDER_KEY_MAP[provider]}'
      )

    model_status = self._fetch_all_models(
        selected_providers={provider}, model_size=model_size,
        output_format=output_format, input_format=input_format,
        feature_tags=feature_tag_list, tool_tags=tool_tag_list,
        raw_config_results_without_test=True, recommended_only=recommended_only
    )

    return self._format_set(model_status.unprocessed_models)

  def get_model(
      self,
      provider: str,
      model: str,
  ) -> types.ProviderModelType:
    """Get a specific model by provider and model name."""
    provider_model_config = (
        self.model_configs_instance.get_provider_model_config((provider, model))
    )
    provider_model = provider_model_config.provider_model

    self.api_key_manager.load_provider_keys()
    if provider_model.provider not in self.api_key_manager.providers_with_key:
      raise ValueError(
          'Provider key not found in environment variables for '
          f'{provider_model.provider}.\n'
          'Required keys: '
          f'{model_configs.PROVIDER_KEY_MAP[provider_model.provider]}'
      )

    return provider_model

  def get_model_config(
      self,
      provider: str,
      model: str,
  ) -> types.ProviderModelConfig:
    """Get the full config for a specific model."""
    return self.model_configs_instance.get_provider_model_config(
        (provider, model)
    )

  def list_working_models(
      self, model_size: types.ModelSizeIdentifierType | None = None,
      verbose: bool = True,
      return_all: bool = False, clear_model_cache: bool = False,
      output_format: types.OutputFormatTypeParam = (
          types.OutputFormatType.TEXT
      ), recommended_only: bool = True
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """List models verified to be working through API tests."""
    if model_size is not None:
      model_size = type_utils.check_model_size_identifier_type(model_size)

    model_status: types.ModelStatus | None = None
    if not self.model_cache_manager:
      logging_utils.log_message(
          logging_options=self.logging_options,
          message='Model cache is not enabled. Fetching all models '
          'from providers. This is not ideal for performance.',
          type=types.LoggingType.WARNING
      )
      model_status = self._fetch_all_models(
          model_size=model_size, output_format=output_format,
          verbose=verbose, recommended_only=recommended_only
      )
    elif (clear_model_cache or not self._check_model_cache_path_same()):
      model_status = self._fetch_all_models(
          model_size=model_size, clear_model_cache=clear_model_cache,
          output_format=output_format, verbose=verbose,
          recommended_only=recommended_only
      )
    else:
      model_status = self._fetch_all_models(
          model_size=model_size, output_format=output_format,
          verbose=verbose, recommended_only=recommended_only
      )

    if return_all:
      return model_status
    return self._format_set(model_status.working_models)

  def check_health(
      self,
      verbose: bool = True,
  ) -> types.ModelStatus:
    """Tests all models and reports health status.

    Always clears the model cache and retests every model. Use
    list_working_models() for cached results.

    Args:
        verbose: If True, prints progress and per-model results.

    Returns:
        ModelStatus with working_models, failed_models, and
        provider_queries.
    """
    if verbose:
      print("> Starting to test each model...")
    model_status = self.list_working_models(
        clear_model_cache=True, verbose=verbose, return_all=True
    )
    if verbose:
      self._print_health_check_results(model_status)
    return model_status

  def _print_health_check_results(
      self,
      model_status: types.ModelStatus,
  ) -> None:
    providers = set([m.provider for m in model_status.working_models] +
                    [m.provider for m in model_status.failed_models])
    result_table = {
        provider: {
            "working": [],
            "failed": []
        } for provider in providers
    }
    for model in model_status.working_models:
      result_table[model.provider]["working"].append(model.model)
    for model in model_status.failed_models:
      result_table[model.provider]["failed"].append(model.model)
    print(
        "> Finished testing.\n"
        f"   Registered Providers: {len(providers)}\n"
        f"   Succeeded Models: {len(model_status.working_models)}\n"
        f"   Failed Models: {len(model_status.failed_models)}"
    )
    for provider in sorted(providers):
      print(f"> {provider}:")
      for model in sorted(result_table[provider]["working"]):
        provider_model = self.model_configs_instance.get_provider_model(
            (provider, model)
        )
        duration = (
            model_status.provider_queries[provider_model].result.timestamp.
            response_time
        )
        print(
            f"   [ WORKING | {duration.total_seconds():6.2f}s ]"
            f": {model}"
        )
      for model in sorted(result_table[provider]["failed"]):
        provider_model = self.model_configs_instance.get_provider_model(
            (provider, model)
        )
        duration = (
            model_status.provider_queries[provider_model].result.timestamp.
            response_time
        )
        print(
            f"   [ FAILED  | {duration.total_seconds():6.2f}s ]"
            f": {model}"
        )

  def list_working_providers(
      self, verbose: bool = True, clear_model_cache: bool = False,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True
  ) -> list[str]:
    """List providers with at least one working model."""
    type_utils.create_output_format_type_list(output_format)
    providers_with_key: set[str] | None = None
    if not self.model_cache_manager:
      # For performance, we only load the provider keys if the
      # model cache is not enabled instead of fetching all models.
      self.api_key_manager.load_provider_keys()
      providers_with_key = set(self.api_key_manager.providers_with_key.keys())
    elif (clear_model_cache or not self._check_model_cache_path_same()):
      model_status = self._fetch_all_models(
          verbose=verbose, clear_model_cache=clear_model_cache,
          output_format=output_format, recommended_only=recommended_only
      )
      providers_with_key = {
          model.provider for model in model_status.working_models
      }
    else:
      model_status = self._fetch_all_models(
          verbose=verbose, output_format=output_format,
          recommended_only=recommended_only
      )
      providers_with_key = {
          model.provider for model in model_status.working_models
      }

    return sorted(providers_with_key)

  def list_working_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      verbose: bool = True,
      return_all: bool = False,
      clear_model_cache: bool = False,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """List working models for a specific provider."""
    type_utils.create_output_format_type_list(output_format)
    if model_size is not None:
      model_size = type_utils.check_model_size_identifier_type(model_size)

    provider_models = self.model_configs_instance.get_all_models(
        provider=provider, model_size=model_size,
        recommended_only=recommended_only
    )

    model_status: types.ModelStatus | None = None
    if not self.model_cache_manager:
      self.api_key_manager.load_provider_keys()
      if provider not in self.api_key_manager.providers_with_key:
        raise ValueError(
            f'Provider key not found in environment variables for {provider}.\n'
            f'Required keys: {model_configs.PROVIDER_KEY_MAP[provider]}'
        )
      model_status = types.ModelStatus()
      for provider_model in provider_models:
        model_status.working_models.add(provider_model)
      self._filter_by_model_size(
          model_status, model_size=model_size, recommended_only=recommended_only
      )
    elif (clear_model_cache or not self._check_model_cache_path_same()):
      model_status = self._fetch_all_models(
          selected_providers={provider}, model_size=model_size,
          output_format=output_format, verbose=verbose,
          clear_model_cache=clear_model_cache, recommended_only=recommended_only
      )
    else:
      model_status = self._fetch_all_models(
          selected_providers={provider}, model_size=model_size,
          output_format=output_format, verbose=verbose,
          recommended_only=recommended_only
      )

    if return_all:
      return model_status
    return self._format_set(model_status.working_models)

  def get_working_model(
      self, provider: str, model: str, verbose: bool = False,
      clear_model_cache: bool = False, output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT)
  ) -> types.ProviderModelType:
    """Get a specific model after verifying it works."""
    type_utils.create_output_format_type_list(output_format)
    provider_model_config = (
        self.model_configs_instance.get_provider_model_config((provider, model))
    )

    provider_model = provider_model_config.provider_model

    model_status: types.ModelStatus | None = None
    if not self.model_cache_manager:
      # For performance, we only load the provider keys if the
      # model cache is not enabled instead of fetching all models.
      self.api_key_manager.load_provider_keys()
      if provider_model.provider not in self.api_key_manager.providers_with_key:
        raise ValueError(
            'Provider key not found in environment variables '
            f'for {provider_model.provider}.\n'
            'Required keys: '
            f'{model_configs.PROVIDER_KEY_MAP[provider_model.provider]}'
        )
      return provider_model
    elif (clear_model_cache or not self._check_model_cache_path_same()):
      model_status = self._fetch_all_models(
          selected_provider_models={provider_model}, verbose=verbose,
          clear_model_cache=clear_model_cache
      )
    else:
      model_status = self._fetch_all_models(
          selected_provider_models={provider_model}, verbose=verbose
      )

    if provider_model in model_status.working_models:
      return provider_model

    raise ValueError(
        'Provider model not found in working models: '
        f'({provider}, {model})\n' + 'Logging Record: ' +
        f'{model_status.provider_queries.get(provider_model, "")}'
    )

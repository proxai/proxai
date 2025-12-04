"""ProxAI Client - Main client class for ProxAI operations.

This module provides the ProxAIClient class which encapsulates all ProxAI
functionality in a StateControlled class, enabling:
- State serialization for multiprocessing
- Multiple client instances
- Thread-safe operation (via separate instances)
"""

import copy
import datetime
import os
import tempfile
from typing import Any, Callable, Dict, Optional, Tuple, Union

import platformdirs

import proxai.types as types
import proxai.type_utils as type_utils
import proxai.state_controllers.state_controller as state_controller
import proxai.caching.model_cache as model_cache_module
import proxai.caching.query_cache as query_cache_module
import proxai.connections.proxdash as proxdash_module
import proxai.connections.available_models as available_models_module
import proxai.connectors.model_configs as model_configs_module
import proxai.connectors.model_connector as model_connector_module
import proxai.connectors.model_registry as model_registry
import proxai.stat_types as stat_types
import proxai.experiment.experiment as experiment
import proxai.serializers.type_serializer as type_serializer
import proxai.logging.utils as logging_utils

_PROXAI_CLIENT_STATE_PROPERTY = '_proxai_client_state'


class ProxAIClient(state_controller.StateControlled):
  """Main client for ProxAI operations.

  This class encapsulates all ProxAI functionality, replacing module-level
  globals with a proper StateControlled implementation.

  Basic Usage:
      client = ProxAIClient(cache_path="/tmp/cache")
      result = client.generate_text("Hello, world!")

  Multiprocessing:
      # Main process
      client = ProxAIClient(cache_path="/tmp/cache")
      state = client.get_state()

      # Worker process
      worker_client = ProxAIClient(init_state=state)
      result = worker_client.generate_text("Hello from worker")
  """

  # === Type Hints for Internal Storage ===

  # Core settings
  _run_type: Optional[types.RunType]
  _hidden_run_key: Optional[str]
  _experiment_path: Optional[str]
  _root_logging_path: Optional[str]

  # Getter functions (for compatibility with sub-components that use getters)
  _get_run_type: Optional[Callable[[], types.RunType]]
  _get_experiment_path: Optional[Callable[[], str]]
  _get_logging_options: Optional[Callable[[], types.LoggingOptions]]
  _get_cache_options: Optional[Callable[[], types.CacheOptions]]
  _get_proxdash_options: Optional[Callable[[], types.ProxDashOptions]]

  # Configuration options
  _logging_options: Optional[types.LoggingOptions]
  _cache_options: Optional[types.CacheOptions]
  _proxdash_options: Optional[types.ProxDashOptions]

  # Nested StateControlled managers
  _model_configs: Optional[model_configs_module.ModelConfigs]
  _model_cache_manager: Optional[model_cache_module.ModelCacheManager]
  _query_cache_manager: Optional[query_cache_module.QueryCacheManager]
  _proxdash_connection: Optional[proxdash_module.ProxDashConnection]
  _available_models: Optional[available_models_module.AvailableModels]

  # Behavior flags
  _strict_feature_test: Optional[bool]
  _suppress_provider_errors: Optional[bool]
  _allow_multiprocessing: Optional[bool]
  _model_test_timeout: Optional[int]
  _model_configs_requested_from_proxdash: Optional[bool]

  # Internal state container
  _proxai_client_state: types.ProxAIClientState

  # === Non-State Runtime Objects ===
  # These are NOT serialized; recreated on load
  _model_connectors: Dict[types.ProviderModelType, model_connector_module.ProviderModelConnector]
  _registered_model_connectors: Dict[types.CallType, model_connector_module.ProviderModelConnector]
  _default_model_cache_manager: Optional[model_cache_module.ModelCacheManager]
  _default_model_cache_path: Optional[str]
  _platform_used_for_default_model_cache: bool
  _stats: Dict[stat_types.GlobalStatType, stat_types.RunStats]

  def __init__(
      self,
      experiment_path: Optional[str] = None,
      cache_path: Optional[str] = None,
      cache_options: Optional[types.CacheOptions] = None,
      logging_path: Optional[str] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      proxdash_options: Optional[types.ProxDashOptions] = None,
      allow_multiprocessing: Optional[bool] = True,
      model_test_timeout: Optional[int] = 25,
      strict_feature_test: Optional[bool] = False,
      suppress_provider_errors: Optional[bool] = False,
      run_type: Optional[types.RunType] = None,
      init_state: Optional[types.ProxAIClientState] = None
  ):
    """Initialize a ProxAI client.

    Args:
        experiment_path: Path for experiment tracking
        cache_path: Shorthand for cache_options.cache_path
        cache_options: Full cache configuration
        logging_path: Shorthand for logging_options.logging_path
        logging_options: Full logging configuration
        proxdash_options: ProxDash integration settings
        allow_multiprocessing: Enable multiprocessing for model tests
        model_test_timeout: Timeout for model connectivity tests (seconds)
        strict_feature_test: Raise errors on unsupported features
        suppress_provider_errors: Return errors instead of raising exceptions
        run_type: PRODUCTION or TEST mode
        init_state: Load from serialized state (exclusive with other args)

    Raises:
        ValueError: If init_state is provided with other arguments
        ValueError: If both cache_path and cache_options.cache_path are set
        ValueError: If both logging_path and logging_options.logging_path are set
    """
    # Parent validates init_state exclusivity and calls init_state()
    super().__init__(
        experiment_path=experiment_path,
        cache_options=cache_options,
        logging_options=logging_options,
        proxdash_options=proxdash_options,
        allow_multiprocessing=allow_multiprocessing,
        model_test_timeout=model_test_timeout,
        strict_feature_test=strict_feature_test,
        suppress_provider_errors=suppress_provider_errors,
        run_type=run_type,
        init_state=init_state
    )

    # Initialize non-state runtime objects
    self._model_connectors = {}
    self._registered_model_connectors = {}
    self._default_model_cache_manager = None
    self._default_model_cache_path = None
    self._platform_used_for_default_model_cache = False
    self._stats = {
        stat_types.GlobalStatType.RUN_TIME: stat_types.RunStats(),
        stat_types.GlobalStatType.SINCE_CONNECT: stat_types.RunStats()
    }

    if init_state:
      self.load_state(init_state)
      self._init_default_model_cache_manager()
      self._post_load_init()
    else:
      initial_state = self.get_state()

      # NOTE: Do NOT set _get_* properties here - they're for external getter injection
      # which causes recursion if we point them to self properties.
      # Instead, set them to None so StateControlled uses direct value access.
      self._get_run_type = None
      self._get_experiment_path = None
      self._get_logging_options = None
      self._get_cache_options = None
      self._get_proxdash_options = None

      # Initialize core settings
      self.run_type = run_type if run_type is not None else types.RunType.PRODUCTION
      self.hidden_run_key = experiment.get_hidden_run_key()

      # Process experiment path (preserving original logic)
      self._process_experiment_path(experiment_path)

      # Process logging options (preserving original logic exactly)
      self._process_logging_options(
          experiment_path=experiment_path,
          logging_path=logging_path,
          logging_options=logging_options
      )

      # Process cache options (preserving original logic exactly)
      self._process_cache_options(
          cache_path=cache_path,
          cache_options=cache_options
      )

      # Process proxdash options (preserving original logic exactly)
      self._process_proxdash_options(proxdash_options=proxdash_options)

      # Set behavior flags
      self.allow_multiprocessing = allow_multiprocessing if allow_multiprocessing is not None else True
      self.model_test_timeout = model_test_timeout if model_test_timeout is not None else 25
      self.strict_feature_test = strict_feature_test if strict_feature_test is not None else False
      self.suppress_provider_errors = suppress_provider_errors if suppress_provider_errors is not None else False
      self.model_configs_requested_from_proxdash = False

      # Initialize managers
      self._init_default_model_cache_manager()
      self._model_configs = model_configs_module.ModelConfigs()
      self._model_cache_manager = None
      self._query_cache_manager = None
      self._proxdash_connection = None
      self._available_models = None

      self.handle_changes(initial_state, self.get_state())

  # === Abstract Method Implementations ===

  def get_internal_state_property_name(self) -> str:
    return _PROXAI_CLIENT_STATE_PROPERTY

  def get_internal_state_type(self):
    return types.ProxAIClientState

  def handle_changes(
      self,
      old_state: types.ProxAIClientState,
      current_state: types.ProxAIClientState
  ):
    """Handle state changes and validate configuration.

    This method is called after state changes to:
    1. Validate the new state
    2. Apply derived changes
    3. Raise errors for invalid configurations
    """
    result_state = copy.deepcopy(old_state)

    # Merge changes from current_state
    if current_state.run_type is not None:
      result_state.run_type = current_state.run_type
    if current_state.hidden_run_key is not None:
      result_state.hidden_run_key = current_state.hidden_run_key
    if current_state.experiment_path is not None:
      result_state.experiment_path = current_state.experiment_path
    if current_state.root_logging_path is not None:
      result_state.root_logging_path = current_state.root_logging_path
    if current_state.logging_options is not None:
      result_state.logging_options = current_state.logging_options
    if current_state.cache_options is not None:
      result_state.cache_options = current_state.cache_options
    if current_state.proxdash_options is not None:
      result_state.proxdash_options = current_state.proxdash_options
    if current_state.strict_feature_test is not None:
      result_state.strict_feature_test = current_state.strict_feature_test
    if current_state.suppress_provider_errors is not None:
      result_state.suppress_provider_errors = current_state.suppress_provider_errors
    if current_state.allow_multiprocessing is not None:
      result_state.allow_multiprocessing = current_state.allow_multiprocessing
    if current_state.model_test_timeout is not None:
      result_state.model_test_timeout = current_state.model_test_timeout
    if current_state.model_configs_requested_from_proxdash is not None:
      result_state.model_configs_requested_from_proxdash = current_state.model_configs_requested_from_proxdash

    # Validation - exactly as in original
    if result_state.model_test_timeout is not None:
      if result_state.model_test_timeout < 1:
        raise ValueError('model_test_timeout must be greater than 0.')

  # === Simple Properties ===

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
    self.set_property_value('experiment_path', value)

  @property
  def root_logging_path(self) -> Optional[str]:
    return self.get_property_value('root_logging_path')

  @root_logging_path.setter
  def root_logging_path(self, value: Optional[str]):
    self.set_property_value('root_logging_path', value)

  @property
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, value: types.LoggingOptions):
    self.set_property_value('logging_options', value)

  @property
  def cache_options(self) -> types.CacheOptions:
    return self.get_property_value('cache_options')

  @cache_options.setter
  def cache_options(self, value: types.CacheOptions):
    self.set_property_value('cache_options', value)

  @property
  def proxdash_options(self) -> types.ProxDashOptions:
    return self.get_property_value('proxdash_options')

  @proxdash_options.setter
  def proxdash_options(self, value: types.ProxDashOptions):
    self.set_property_value('proxdash_options', value)

  @property
  def strict_feature_test(self) -> bool:
    return self.get_property_value('strict_feature_test')

  @strict_feature_test.setter
  def strict_feature_test(self, value: bool):
    self.set_property_value('strict_feature_test', value)

  @property
  def suppress_provider_errors(self) -> bool:
    return self.get_property_value('suppress_provider_errors')

  @suppress_provider_errors.setter
  def suppress_provider_errors(self, value: bool):
    self.set_property_value('suppress_provider_errors', value)

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
  def model_configs_requested_from_proxdash(self) -> bool:
    return self.get_property_value('model_configs_requested_from_proxdash')

  @model_configs_requested_from_proxdash.setter
  def model_configs_requested_from_proxdash(self, value: bool):
    self.set_property_value('model_configs_requested_from_proxdash', value)

  # === Nested StateControlled Properties ===

  @property
  def model_configs(self) -> model_configs_module.ModelConfigs:
    return self.get_state_controlled_property_value('model_configs')

  @model_configs.setter
  def model_configs(self, value):
    self.set_state_controlled_property_value('model_configs', value)

  def model_configs_deserializer(
      self,
      state_value: types.ModelConfigsState
  ) -> model_configs_module.ModelConfigs:
    return model_configs_module.ModelConfigs(init_state=state_value)

  @property
  def model_cache_manager(self) -> Optional[model_cache_module.ModelCacheManager]:
    return self.get_state_controlled_property_value('model_cache_manager')

  @model_cache_manager.setter
  def model_cache_manager(self, value):
    self.set_state_controlled_property_value('model_cache_manager', value)

  def model_cache_manager_deserializer(
      self,
      state_value: types.ModelCacheManagerState
  ) -> model_cache_module.ModelCacheManager:
    return model_cache_module.ModelCacheManager(init_state=state_value)

  @property
  def query_cache_manager(self) -> Optional[query_cache_module.QueryCacheManager]:
    return self.get_state_controlled_property_value('query_cache_manager')

  @query_cache_manager.setter
  def query_cache_manager(self, value):
    self.set_state_controlled_property_value('query_cache_manager', value)

  def query_cache_manager_deserializer(
      self,
      state_value: types.QueryCacheManagerState
  ) -> query_cache_module.QueryCacheManager:
    return query_cache_module.QueryCacheManager(init_state=state_value)

  @property
  def proxdash_connection(self) -> Optional[proxdash_module.ProxDashConnection]:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value):
    self.set_state_controlled_property_value('proxdash_connection', value)

  def proxdash_connection_deserializer(
      self,
      state_value: types.ProxDashConnectionState
  ) -> proxdash_module.ProxDashConnection:
    return proxdash_module.ProxDashConnection(init_state=state_value)

  @property
  def available_models(self) -> Optional[available_models_module.AvailableModels]:
    return self.get_state_controlled_property_value('available_models')

  @available_models.setter
  def available_models(self, value):
    self.set_state_controlled_property_value('available_models', value)

  def available_models_deserializer(
      self,
      state_value: types.AvailableModelsState
  ) -> available_models_module.AvailableModels:
    return available_models_module.AvailableModels(init_state=state_value)

  # === Private Helper Methods - Option Processing ===
  # These preserve the exact business logic from the original _set_* functions

  def _process_experiment_path(
      self,
      experiment_path: Optional[str] = None
  ) -> Optional[str]:
    """Process and validate experiment path.

    Preserves exact logic from original _set_experiment_path.
    """
    if experiment_path is None:
      self.experiment_path = None
      return None
    experiment.validate_experiment_path(experiment_path)
    self.experiment_path = experiment_path
    return experiment_path

  def _process_logging_options(
      self,
      experiment_path: Optional[str] = None,
      logging_path: Optional[str] = None,
      logging_options: Optional[types.LoggingOptions] = None
  ) -> Tuple[types.LoggingOptions, Optional[str]]:
    """Process logging options.

    Preserves exact logic from original _set_logging_options.
    """
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

    self.root_logging_path = root_logging_path
    self.logging_options = result_logging_options

    return (result_logging_options, root_logging_path)

  def _process_cache_options(
      self,
      cache_path: Optional[str] = None,
      cache_options: Optional[types.CacheOptions] = None
  ) -> types.CacheOptions:
    """Process cache options.

    Preserves exact logic from original _set_cache_options.
    """
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

    self.cache_options = result_cache_options
    return result_cache_options

  def _process_proxdash_options(
      self,
      proxdash_options: Optional[types.ProxDashOptions] = None
  ) -> types.ProxDashOptions:
    """Process proxdash options.

    Preserves exact logic from original _set_proxdash_options.
    """
    result_proxdash_options = types.ProxDashOptions()
    if proxdash_options is not None:
      result_proxdash_options.stdout = proxdash_options.stdout
      result_proxdash_options.hide_sensitive_content = (
          proxdash_options.hide_sensitive_content)
      result_proxdash_options.disable_proxdash = proxdash_options.disable_proxdash
      result_proxdash_options.api_key = proxdash_options.api_key
      result_proxdash_options.base_url = proxdash_options.base_url

    self.proxdash_options = result_proxdash_options
    return result_proxdash_options

  # === Private Helper Methods - Initialization ===

  def _init_default_model_cache_manager(self):
    """Initialize the default model cache manager.

    Preserves exact logic from original _init_default_model_cache_manager.
    Uses platform-specific cache directory when possible,
    falls back to temporary directory otherwise.
    """
    try:
      app_dirs = platformdirs.PlatformDirs(appname="proxai", appauthor="proxai")
      self._default_model_cache_path = app_dirs.user_cache_dir
      os.makedirs(self._default_model_cache_path, exist_ok=True)
      # 4 hours cache duration makes sense for local development if proxai is
      # using platform app cache directory
      self._default_model_cache_manager = model_cache_module.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=self._default_model_cache_path,
              model_cache_duration=60 * 60 * 4))
      self._platform_used_for_default_model_cache = True
    except Exception as e:
      temp_dir = tempfile.TemporaryDirectory()
      self._default_model_cache_path = temp_dir.name
      self._default_model_cache_manager = model_cache_module.ModelCacheManager(
          cache_options=types.CacheOptions(
              cache_path=self._default_model_cache_path))
      self._platform_used_for_default_model_cache = False

  def _post_load_init(self):
    """Initialize runtime objects after loading state.

    Called after load_state() to set up objects that aren't serialized.
    """
    # Set getter functions to None - we use direct property access
    self._get_run_type = None
    self._get_experiment_path = None
    self._get_logging_options = None
    self._get_cache_options = None
    self._get_proxdash_options = None

    # Initialize model_configs if not loaded from state
    if self._model_configs is None:
      self._model_configs = model_configs_module.ModelConfigs()

  # === Getter Methods for Sub-Components ===
  # These methods are passed to sub-components as callbacks.
  # They use get_property_internal_value to avoid the StateControlled getter chain.

  def _getter_run_type(self) -> types.RunType:
    """Getter for sub-components."""
    return self.get_property_internal_value('run_type')

  def _getter_experiment_path(self) -> Optional[str]:
    """Getter for sub-components."""
    return self.get_property_internal_value('experiment_path')

  def _getter_logging_options(self) -> types.LoggingOptions:
    """Getter for sub-components."""
    return self.get_property_internal_value('logging_options')

  def _getter_cache_options(self) -> types.CacheOptions:
    """Getter for sub-components."""
    return self.get_property_internal_value('cache_options')

  def _getter_proxdash_options(self) -> types.ProxDashOptions:
    """Getter for sub-components."""
    return self.get_property_internal_value('proxdash_options')

  # === Private Helper Methods - Lazy Getters ===
  # These preserve the exact lazy initialization logic from the original

  def _get_model_configs_instance(self) -> model_configs_module.ModelConfigs:
    """Get model configs, fetching from proxdash if needed.

    Preserves exact logic from original _get_model_configs.
    """
    if not self.model_configs_requested_from_proxdash:
      model_configs_schema = self._get_proxdash_connection_instance().get_model_configs_schema()
      if model_configs_schema is not None:
        self._model_configs.model_configs_schema = model_configs_schema
      self.model_configs_requested_from_proxdash = True
    return self._model_configs

  def _get_model_cache_manager_instance(self) -> model_cache_module.ModelCacheManager:
    """Get model cache manager with fallback to default.

    Preserves exact logic from original _get_model_cache_manager.
    """
    if self._model_cache_manager is None:
      self._model_cache_manager = model_cache_module.ModelCacheManager(
          get_cache_options=self._getter_cache_options)
    if (self._model_cache_manager.status !=
        types.ModelCacheManagerStatus.CACHE_PATH_NOT_FOUND):
      return self._model_cache_manager

    if self._default_model_cache_path is not None:
      return self._default_model_cache_manager

    raise ValueError('Model cache manager is not initialized and there is no '
                     'default model cache manager.')

  def _get_query_cache_manager_instance(self) -> query_cache_module.QueryCacheManager:
    """Get query cache manager.

    Preserves exact logic from original _get_query_cache_manager.
    """
    if self._query_cache_manager is None:
      self._query_cache_manager = query_cache_module.QueryCacheManager(
          get_cache_options=self._getter_cache_options)

    return self._query_cache_manager

  def _get_proxdash_connection_instance(self) -> proxdash_module.ProxDashConnection:
    """Get proxdash connection.

    Preserves exact logic from original _get_proxdash_connection.
    """
    if self._proxdash_connection is None:
      self._proxdash_connection = proxdash_module.ProxDashConnection(
          hidden_run_key=self.hidden_run_key,
          get_experiment_path=self._getter_experiment_path,
          get_logging_options=self._getter_logging_options,
          get_proxdash_options=self._getter_proxdash_options)
    return self._proxdash_connection

  def _get_model_connector(
      self,
      provider_model_identifier: types.ProviderModelIdentifierType
  ) -> model_connector_module.ProviderModelConnector:
    """Get or create a model connector.

    Preserves exact logic from original _get_model_connector.
    """
    model_configs_instance = self._get_model_configs_instance()
    provider_model = model_configs_instance.get_provider_model(
        provider_model_identifier)
    if provider_model in self._model_connectors:
      return self._model_connectors[provider_model]

    connector = model_registry.get_model_connector(
        provider_model,
        model_configs=self._get_model_configs_instance())
    self._model_connectors[provider_model] = connector(
        get_run_type=self._getter_run_type,
        get_strict_feature_test=lambda: self.get_property_internal_value('strict_feature_test'),
        get_query_cache_manager=self._get_query_cache_manager_instance,
        get_logging_options=self._getter_logging_options,
        get_proxdash_connection=self._get_proxdash_connection_instance,
        stats=self._stats)
    return self._model_connectors[provider_model]

  def _get_registered_model_connector(
      self,
      call_type: types.CallType = types.CallType.GENERATE_TEXT
  ) -> model_connector_module.ProviderModelConnector:
    """Get or create a registered model connector.

    Preserves exact logic from original _get_registered_model_connector.
    """
    if call_type not in self._registered_model_connectors:
      if call_type == types.CallType.GENERATE_TEXT:
        avail_models = self.get_available_models()
        if (
            not avail_models.model_cache_manager or
            not avail_models.model_cache_manager.get(
                types.CallType.GENERATE_TEXT).working_models):
          print('Checking available models, this may take a while...')
        models = self.get_available_models().list_models(return_all=True)
        model_configs_instance = self._get_model_configs_instance()
        for provider_model in model_configs_instance.get_default_model_priority_list():
          if provider_model in models.working_models:
            self._registered_model_connectors[call_type] = self._get_model_connector(
                provider_model)
            break
        if call_type not in self._registered_model_connectors:
          if models.working_models:
            self._registered_model_connectors[call_type] = self._get_model_connector(
                models.working_models.pop())
          else:
            raise ValueError(
                'No working models found in current environment:\n'
                '* Please check your environment variables and try again.\n'
                '* You can use px.check_health() method as instructed in '
                'https://www.proxai.co/proxai-docs/check-health')
    return self._registered_model_connectors[call_type]

  # === Public API Methods ===

  def connect(
      self,
      experiment_path: Optional[str] = None,
      cache_path: Optional[str] = None,
      cache_options: Optional[types.CacheOptions] = None,
      logging_path: Optional[str] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      proxdash_options: Optional[types.ProxDashOptions] = None,
      allow_multiprocessing: Optional[bool] = True,
      model_test_timeout: Optional[int] = 25,
      strict_feature_test: Optional[bool] = False,
      suppress_provider_errors: Optional[bool] = False
  ):
    """Reconfigure the client with new options.

    Preserves exact logic from original connect function.
    """
    self._process_experiment_path(experiment_path=experiment_path)
    self._process_logging_options(
        experiment_path=experiment_path,
        logging_path=logging_path,
        logging_options=logging_options)
    self._process_cache_options(
        cache_path=cache_path,
        cache_options=cache_options)
    self._process_proxdash_options(proxdash_options=proxdash_options)

    if allow_multiprocessing is not None:
      self.allow_multiprocessing = allow_multiprocessing
    if model_test_timeout is not None:
      if model_test_timeout < 1:
        raise ValueError('model_test_timeout must be greater than 0.')
      self.model_test_timeout = model_test_timeout
    self.model_configs_requested_from_proxdash = False
    if strict_feature_test is not None:
      self.strict_feature_test = strict_feature_test
    if suppress_provider_errors is not None:
      self.suppress_provider_errors = suppress_provider_errors

    # This ensures updating model cache manager instead of default model cache
    # manager.
    if self._model_cache_manager is not None:
      self._model_cache_manager.apply_external_state_changes()

    model_cache_mgr = self._get_model_cache_manager_instance()
    query_cache_mgr = self._get_query_cache_manager_instance()
    proxdash_conn = self._get_proxdash_connection_instance()

    query_cache_mgr.apply_external_state_changes()
    proxdash_conn.apply_external_state_changes()

    for connector in self._model_connectors.values():
      connector.apply_external_state_changes()

    cache_opts = self.cache_options
    if cache_opts.clear_model_cache_on_connect:
      model_cache_mgr.clear_cache()
    if cache_opts.clear_query_cache_on_connect:
      query_cache_mgr.clear_cache()

  def generate_text(
      self,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = None,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None,
      provider_model: Optional[types.ProviderModelIdentifierType] = None,
      use_cache: Optional[bool] = None,
      unique_response_limit: Optional[int] = None,
      extensive_return: bool = False,
      suppress_provider_errors: Optional[bool] = None
  ) -> Union[str, types.LoggingRecord]:
    """Generate text using the configured model.

    Preserves exact logic from original generate_text function.
    """
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages is not None:
      type_utils.check_messages_type(messages)

    if use_cache:
      query_cache_mgr = self._get_query_cache_manager_instance()
      if query_cache_mgr.status != types.QueryCacheManagerStatus.WORKING:
        raise ValueError(
            'use_cache is True but query cache is not working.\n'
            f'Query Cache Manager Status: {query_cache_mgr.status}')
    elif use_cache is None:
      query_cache_mgr = self._get_query_cache_manager_instance()
      use_cache = (
          query_cache_mgr.status == types.QueryCacheManagerStatus.WORKING)

    if provider_model is not None:
      connector = self._get_model_connector(
          provider_model_identifier=provider_model)
    else:
      connector = self._get_registered_model_connector(
          call_type=types.CallType.GENERATE_TEXT)

    logging_record: types.LoggingRecord = connector.generate_text(
        prompt=prompt,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        use_cache=use_cache,
        unique_response_limit=unique_response_limit)
    if logging_record.response_record.error:
      if suppress_provider_errors or (
          suppress_provider_errors is None and self.suppress_provider_errors):
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

  def set_model(
      self,
      provider_model: Optional[types.ProviderModelIdentifierType] = None,
      generate_text: Optional[types.ProviderModelIdentifierType] = None
  ):
    """Set the model to use for generation.

    Preserves exact logic from original set_model function.
    """
    if provider_model and generate_text:
      raise ValueError('provider_model and generate_text cannot be set at the '
                       'same time. Please set one of them.')

    if provider_model is None and generate_text is None:
      raise ValueError('provider_model or generate_text must be set.')

    if generate_text:
      provider_model = generate_text

    model_configs_instance = self._get_model_configs_instance()
    model_configs_instance.check_provider_model_identifier_type(provider_model)
    self._registered_model_connectors[
        types.CallType.GENERATE_TEXT] = self._get_model_connector(provider_model)

  def get_summary(
      self,
      run_time: bool = False,
      json: bool = False
  ) -> Union[stat_types.RunStats, Dict[str, Any]]:
    """Get usage statistics.

    Preserves exact logic from original get_summary function.
    """
    stat_value = None
    if run_time:
      stat_value = copy.deepcopy(self._stats[stat_types.GlobalStatType.RUN_TIME])
    else:
      stat_value = copy.deepcopy(self._stats[stat_types.GlobalStatType.SINCE_CONNECT])

    if json:
      return type_serializer.encode_run_stats(stat_value)

    class StatValue(stat_types.RunStats):
      def __init__(self, stat_value):
        super().__init__(**stat_value.__dict__)

      def serialize(self):
        return type_serializer.encode_run_stats(self)

    return StatValue(stat_value)

  def get_available_models(self) -> available_models_module.AvailableModels:
    """Get available models.

    Preserves exact logic from original get_available_models function.
    """
    if self._available_models is None:
      self._available_models = available_models_module.AvailableModels(
          get_run_type=self._getter_run_type,
          get_model_configs=self._get_model_configs_instance,
          get_model_connector=self._get_model_connector,
          get_allow_multiprocessing=lambda: self.get_property_internal_value('allow_multiprocessing'),
          get_model_test_timeout=lambda: self.get_property_internal_value('model_test_timeout'),
          get_logging_options=self._getter_logging_options,
          get_model_cache_manager=self._get_model_cache_manager_instance,
          get_proxdash_connection=self._get_proxdash_connection_instance)
    return self._available_models

  def get_current_options(
      self,
      json: bool = False
  ) -> Union[types.RunOptions, Dict[str, Any]]:
    """Get current configuration options.

    Preserves exact logic from original get_current_options function.
    """
    run_options = types.RunOptions(
        run_type=self.run_type,
        hidden_run_key=self.hidden_run_key,
        experiment_path=self.experiment_path,
        root_logging_path=self.root_logging_path,
        default_model_cache_path=self._default_model_cache_path,
        logging_options=self.logging_options,
        cache_options=self.cache_options,
        proxdash_options=self.proxdash_options,
        strict_feature_test=self.strict_feature_test,
        suppress_provider_errors=self.suppress_provider_errors,
        allow_multiprocessing=self.allow_multiprocessing,
        model_test_timeout=self.model_test_timeout)
    if json:
      return type_serializer.encode_run_options(run_options=run_options)
    return run_options

  def reset_platform_cache(self):
    """Reset platform cache.

    Preserves exact logic from original reset_platform_cache function.
    """
    if self._platform_used_for_default_model_cache and self._default_model_cache_manager:
      self._default_model_cache_manager.clear_cache()

  def check_health(
      self,
      experiment_path: Optional[str] = None,
      verbose: bool = True,
      allow_multiprocessing: bool = True,
      model_test_timeout: int = 25,
      extensive_return: bool = False,
  ) -> types.ModelStatus:
    """Check connectivity to all configured models.

    Preserves exact logic from original check_health function.
    """
    if experiment_path is None:
      if self.experiment_path is None:
        now = datetime.datetime.now()
        experiment_path = (
            f'connection_health/{now.strftime("%Y-%m-%d_%H-%M-%S")}')
        experiment.validate_experiment_path(experiment_path)
        logging_opts, _ = self._process_logging_options(
            experiment_path=experiment_path,
            logging_options=self.logging_options)
      else:
        experiment_path = self.experiment_path
        logging_opts = self.logging_options
    else:
      experiment.validate_experiment_path(experiment_path)
      logging_opts, _ = self._process_logging_options(
          experiment_path=experiment_path,
          logging_options=self.logging_options)

    if self.run_type == types.RunType.TEST:
      proxdash_opts = types.ProxDashOptions(
          stdout=False,
          disable_proxdash=True)
    else:
      proxdash_opts = copy.deepcopy(self.proxdash_options)
      proxdash_opts.stdout = verbose

    model_configs_instance = self._get_model_configs_instance()

    proxdash_conn = proxdash_module.ProxDashConnection(
        hidden_run_key=self.hidden_run_key,
        experiment_path=experiment_path,
        logging_options=logging_opts,
        proxdash_options=proxdash_opts)

    def _get_modified_model_connector(
        provider_model_identifier: types.ProviderModelIdentifierType
    ) -> model_connector_module.ProviderModelConnector:
      provider_model = model_configs_instance.get_provider_model(
          provider_model_identifier)
      connector = model_registry.get_model_connector(
          provider_model,
          model_configs=self._get_model_configs_instance())
      return connector(
          get_run_type=self._getter_run_type,
          get_strict_feature_test=lambda: self.get_property_internal_value('strict_feature_test'),
          get_query_cache_manager=self._get_query_cache_manager_instance,
          logging_options=logging_opts,
          proxdash_connection=proxdash_conn,
          stats=self._stats)

    if verbose:
      print('> Starting to test each model...')
    models = available_models_module.AvailableModels(
        run_type=self.run_type,
        model_configs=self._get_model_configs_instance(),
        logging_options=logging_opts,
        proxdash_connection=proxdash_conn,
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
    if proxdash_conn.status == types.ProxDashConnectionStatus.CONNECTED:
      logging_utils.log_proxdash_message(
          logging_options=logging_opts,
          proxdash_options=proxdash_opts,
          message='Results are uploaded to the ProxDash.',
          type=types.LoggingType.INFO)
    if extensive_return:
      return model_status

  # === State Export/Import ===

  def export_state(self) -> types.ProxAIClientState:
    """Export current state for serialization.

    Use this to pass state to worker processes.

    Returns:
        ProxAIClientState that can be serialized and passed to workers.
    """
    return self.get_state()

  @classmethod
  def from_state(cls, state: types.ProxAIClientState) -> 'ProxAIClient':
    """Create a client from exported state.

    Use this in worker processes to restore state.

    Args:
        state: State previously exported via export_state()

    Returns:
        New ProxAIClient instance with restored state.
    """
    return cls(init_state=state)

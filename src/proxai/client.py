import dataclasses
import copy
import inspect
import os
import pydantic
import tempfile
from collections.abc import Callable
from typing import Dict, Any, List

import platformdirs

import proxai.chat.chat_session as chat_session
import proxai.caching.model_cache as model_cache
import proxai.caching.query_cache as query_cache
import proxai.connections.available_models as available_models
import proxai.connections.proxdash as proxdash
import proxai.connectors.model_configs as model_configs
import proxai.experiment.experiment as experiment
import proxai.logging.utils as logging_utils
import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller
import proxai.type_utils as type_utils
import proxai.types as types

_PROXAI_CLIENT_STATE_PROPERTY = "_proxai_client_state"


class ModelConnector:
  """Provides access to model discovery and availability information.

  This class offers methods to list available models, providers, and check
  which models are currently working. It can be accessed via the ``px.models``
  singleton for the default client, or via ``client.models`` for a specific
  client instance.

  Example:
      >>> import proxai as px
      >>> # Using the default client singleton
      >>> models = px.models.list_models()
      >>> # Using a specific client instance
      >>> client = px.Client()
      >>> models = client.models.list_models()
  """

  def __init__(
      self,
      client_getter: Callable[[], "ProxAIClient"],
  ) -> None:
    """Initializes the ModelConnector with a client getter function.

    Args:
        client_getter: A callable that returns the ProxAIClient instance
            to use for model operations. This allows the connector to work
            with both the default global client and specific client instances.
    """
    self._client_getter = client_getter

  def list_models(
      self,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureTagParam | None = None,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType]:
    """Lists all configured models matching the specified criteria.

    Returns models that are configured in the system regardless of whether
    they are currently accessible or working. Use list_working_models()
    to get only models that have been verified to work.

    Args:
        model_size: Filter by model size category. Can be a ModelSizeType
            enum value ('small', 'medium', 'large', 'largest') or a string.
        features: Filter by required features. List of feature names that
            models must support (e.g., ['system', 'temperature']).
        call_type: The type of API call to filter models for. Can be a
            CallType enum or a string (e.g., 'text', 'image').
            Defaults to TEXT.
        recommended_only: If True, returns only recommended models curated
            by ProxAI. Set to False to include all available models.
            Defaults to True.

    Returns:
        list[types.ProviderModelType]: A list of ProviderModelType objects
            representing the matching models.

    Example:
        >>> import proxai as px
        >>> # List all models
        >>> models = px.models.list_models()
        >>> print(models[0])
        (openai, gpt-4)

        >>> # Filter by size
        >>> large_models = px.models.list_models(model_size="large")

        >>> # Using a specific client
        >>> client = px.Client()
        >>> models = client.models.list_models()
    """
    return self._client_getter().available_models_instance.list_models(
        model_size=model_size,
        features=features,
        call_type=call_type,
        recommended_only=recommended_only,
    )

  def list_providers(
      self,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[str]:
    """Lists all providers that have API keys configured.

    Returns provider names for which the required environment variables
    are set, indicating the provider can potentially be used.

    Args:
        call_type: The type of API call to filter providers for. Can be a
            CallType enum or a string (e.g., 'text', 'image').
            Defaults to TEXT.
        recommended_only: If True, returns only providers with recommended
            models curated by ProxAI. Set to False to include all
            available providers. Defaults to True.

    Returns:
        List[str]: A sorted list of provider names (e.g., ['anthropic',
            'openai']).

    Example:
        >>> import proxai as px
        >>> providers = px.models.list_providers()
        >>> print(providers)
        ['anthropic', 'google', 'openai']

        >>> # Using a specific client
        >>> client = px.Client()
        >>> providers = client.models.list_providers()
    """
    return self._client_getter().available_models_instance.list_providers(
        call_type=call_type,
        recommended_only=recommended_only,
    )

  def list_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureTagParam | None = None,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType]:
    """Lists all models available from a specific provider.

    Returns models for the given provider that match the specified
    filtering criteria.

    Args:
        provider: The provider name to list models for (e.g., 'openai',
            'anthropic').
        model_size: Filter by model size category. Can be a ModelSizeType
            enum value or a string.
        features: Filter by required features. List of feature names that
            models must support.
        call_type: The type of API call to filter models for. Can be a
            CallType enum or a string (e.g., 'text', 'image').
            Defaults to TEXT.
        recommended_only: If True, returns only recommended models curated
            by ProxAI. Set to False to include all available models.
            Defaults to True.

    Returns:
        list[types.ProviderModelType]: A list of ProviderModelType objects
            representing the matching models for the provider.

    Raises:
        ValueError: If the provider's API key is not found in environment
            variables.

    Example:
        >>> import proxai as px
        >>> openai_models = px.models.list_provider_models("openai")
        >>> print(openai_models)
        [(openai, gpt-4), (openai, gpt-3.5-turbo), ...]

        >>> # Using a specific client
        >>> client = px.Client()
        >>> openai_models = client.models.list_provider_models("openai")
    """
    return self._client_getter().available_models_instance.list_provider_models(
        provider=provider,
        model_size=model_size,
        features=features,
        call_type=call_type,
        recommended_only=recommended_only,
    )

  def get_model(
      self,
      provider: str,
      model: str,
  ) -> types.ProviderModelType:
    """Gets a specific model by provider and model name.

    Returns the ProviderModelType for the specified provider and model
    combination if it exists and the provider's API key is configured.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model: The model name (e.g., 'gpt-4', 'claude-3-opus').

    Returns:
        types.ProviderModelType: The model information including provider,
            model name, and provider-specific identifier.

    Raises:
        ValueError: If the provider's API key is not found, or if the
            model doesn't exist.

    Example:
        >>> import proxai as px
        >>> model = px.models.get_model("openai", "gpt-4")
        >>> print(model)
        (openai, gpt-4)

        >>> # Using a specific client
        >>> client = px.Client()
        >>> model = client.models.get_model("openai", "gpt-4")
    """
    return self._client_getter().available_models_instance.get_model(
        provider=provider,
        model=model,
    )

  def list_working_models(
      self,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureTagParam | None = None,
      verbose: bool = True,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """Lists models that have been verified to be working.

    Tests each configured model and returns only those that successfully
    respond. Results are cached to avoid repeated testing.

    Args:
        model_size: Filter by model size category.
        features: Filter by required features.
        verbose: If True, prints progress information during testing.
            Defaults to True.
        return_all: If True, returns a ModelStatus object with working,
            failed, and filtered models. Defaults to False.
        clear_model_cache: If True, clears the model cache and retests
            all models. Defaults to False.
        call_type: The type of API call to test. Can be a CallType enum
            or a string (e.g., 'text'). Defaults to TEXT.
        recommended_only: If True, returns only recommended models curated
            by ProxAI. Set to False to include all available models.
            Defaults to True.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            working ProviderModelType objects, or a ModelStatus object if
            return_all is True.

    Example:
        >>> import proxai as px
        >>> working_models = px.models.list_working_models(verbose=False)
        >>> print(f"Found {len(working_models)} working models")

        >>> # Using a specific client
        >>> client = px.Client()
        >>> working_models = client.models.list_working_models(verbose=False)
    """
    return self._client_getter().available_models_instance.list_working_models(
        model_size=model_size,
        features=features,
        verbose=verbose,
        return_all=return_all,
        clear_model_cache=clear_model_cache,
        call_type=call_type,
        recommended_only=recommended_only,
    )

  def list_working_providers(
      self,
      verbose: bool = True,
      clear_model_cache: bool = False,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[str]:
    """Lists providers that have at least one working model.

    Tests models and returns providers that have successfully responded
    to at least one test request.

    Args:
        verbose: If True, prints progress information during testing.
            Defaults to True.
        clear_model_cache: If True, clears the model cache and retests
            all models. Defaults to False.
        call_type: The type of API call to test. Can be a CallType enum
            or a string (e.g., 'text'). Defaults to TEXT.
        recommended_only: If True, returns only providers with recommended
            models curated by ProxAI. Set to False to include all
            available providers. Defaults to True.

    Returns:
        List[str]: A sorted list of provider names with working models.

    Example:
        >>> import proxai as px
        >>> providers = px.models.list_working_providers(verbose=False)
        >>> print(providers)
        ['anthropic', 'openai']

        >>> # Using a specific client
        >>> client = px.Client()
        >>> providers = client.models.list_working_providers(verbose=False)
    """
    return (
        self._client_getter().available_models_instance.list_working_providers(
            verbose=verbose,
            clear_model_cache=clear_model_cache,
            call_type=call_type,
            recommended_only=recommended_only,
        )
    )

  def list_working_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureTagParam | None = None,
      verbose: bool = True,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallTypeParam = types.CallType.TEXT,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """Lists working models from a specific provider.

    Tests models from the specified provider and returns only those that
    successfully respond.

    Args:
        provider: The provider name to list models for.
        model_size: Filter by model size category.
        features: Filter by required features.
        verbose: If True, prints progress information during testing.
            Defaults to True.
        return_all: If True, returns a ModelStatus object with detailed
            results. Defaults to False.
        clear_model_cache: If True, clears the model cache and retests
            models. Defaults to False.
        call_type: The type of API call to test. Can be a CallType enum
            or a string (e.g., 'text'). Defaults to TEXT.
        recommended_only: If True, returns only recommended models curated
            by ProxAI. Set to False to include all available models.
            Defaults to True.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            working ProviderModelType objects from the provider, or a
            ModelStatus object if return_all is True.

    Raises:
        ValueError: If the provider's API key is not found.

    Example:
        >>> import proxai as px
        >>> openai_working = px.models.list_working_provider_models(
        ...   "openai", verbose=False
        ... )
        >>> print(openai_working)
        [(openai, gpt-4), (openai, gpt-3.5-turbo)]

        >>> # Using a specific client
        >>> client = px.Client()
        >>> openai_working = client.models.list_working_provider_models(
        ...   "openai", verbose=False
        ... )
    """
    available_models = self._client_getter().available_models_instance
    return available_models.list_working_provider_models(
        provider=provider,
        model_size=model_size,
        features=features,
        verbose=verbose,
        return_all=return_all,
        clear_model_cache=clear_model_cache,
        call_type=call_type,
        recommended_only=recommended_only,
    )

  def get_working_model(
      self,
      provider: str,
      model: str,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallTypeParam = types.CallType.TEXT,
  ) -> types.ProviderModelType:
    """Verifies and returns a specific model if it's working.

    Tests the specified model and returns it only if it successfully
    responds. Results are cached.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model: The model name (e.g., 'gpt-4', 'claude-3-opus').
        verbose: If True, prints progress information during testing.
            Defaults to False.
        clear_model_cache: If True, clears the model cache and retests
            the model. Defaults to False.
        call_type: The type of API call to test. Can be a CallType enum
            or a string (e.g., 'text'). Defaults to TEXT.

    Returns:
        types.ProviderModelType: The model information if the model is
            working.

    Raises:
        ValueError: If the provider's API key is not found, the model
            doesn't exist, or the model failed the health check.

    Example:
        >>> import proxai as px
        >>> model = px.models.get_working_model("openai", "gpt-4")
        >>> print(model)
        (openai, gpt-4)

        >>> # Using a specific client
        >>> client = px.Client()
        >>> model = client.models.get_working_model("openai", "gpt-4")
    """
    return self._client_getter().available_models_instance.get_working_model(
        provider=provider,
        model=model,
        verbose=verbose,
        clear_model_cache=clear_model_cache,
        call_type=call_type,
    )


@dataclasses.dataclass
class ProxAIClientParams:
  """Initialization parameters for ProxAIClient."""

  experiment_path: str | None = None
  cache_options: types.CacheOptions | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_options: types.ProxDashOptions | None = None
  allow_multiprocessing: bool | None = True
  model_test_timeout: int | None = 25
  feature_mapping_strategy: types.FeatureMappingStrategy | None = (
      types.FeatureMappingStrategy.BEST_EFFORT
  )
  suppress_provider_errors: bool | None = False
  keep_raw_provider_response: bool | None = False


class ProxAIClient(state_controller.StateControlled):
  """A client for interacting with multiple AI providers through a unified API.

  ProxAIClient provides a consistent interface for text generation across
  different AI providers (OpenAI, Anthropic, Google, etc.). It handles
  model selection, caching, logging, and error handling.

  This class can be used directly for more control over client instances,
  or accessed via the module-level functions (px.connect(), px.generate_text())
  which use a default global client.

  Example:
      >>> import proxai as px
      >>> # Create a custom client instance
      >>> client = px.Client(
      ...   cache_options=px.CacheOptions(cache_path="/tmp/cache"),
      ...   logging_options=px.LoggingOptions(stdout=True),
      ... )
      >>> response = client.generate_text(prompt="Hello, world!")

      >>> # Or use the simpler module-level API
      >>> px.connect(cache_options=px.CacheOptions(cache_path="/tmp/cache"))
      >>> response = px.generate_text(prompt="Hello, world!")
  """

  def __init__(
      self,
      experiment_path: str | None = None,
      cache_options: types.CacheOptions | None = None,
      logging_options: types.LoggingOptions | None = None,
      proxdash_options: types.ProxDashOptions | None = None,
      allow_multiprocessing: bool | None = True,
      model_test_timeout: int | None = 25,
      feature_mapping_strategy: (types.FeatureMappingStrategy | None
                                ) = types.FeatureMappingStrategy.BEST_EFFORT,
      suppress_provider_errors: bool | None = False,
      keep_raw_provider_response: bool | None = False,
      init_from_params: ProxAIClientParams | None = None,
      init_from_state: types.ProxAIClientState | None = None,
  ) -> None:
    """Initializes a new ProxAI client instance.

    Creates a client with the specified configuration for caching, logging,
    and behavior options. The client can be configured either through
    individual parameters or by passing a ProxAIClientParams object.

    Args:
        experiment_path: Path identifier for organizing experiments. Used to
            group related API calls and logs together under a common path.
        cache_options: Configuration for query and model caching behavior.
            Controls cache paths, response limits, and cache clearing options.
            See CacheOptions for available settings.
        logging_options: Configuration for logging behavior. Controls log file
            paths, stdout output, and sensitive content handling.
            See LoggingOptions for available settings.
        proxdash_options: Configuration for ProxDash monitoring integration.
            Controls API key, base URL, and output options.
            See ProxDashOptions for available settings.
        allow_multiprocessing: Whether to test models in parallel using
            multiprocessing. Disable if you encounter process spawning errors
            (common in Jupyter notebooks, AWS Lambda, or on Windows/macOS
            without proper multiprocessing guards). Defaults to True.
        model_test_timeout: Timeout in seconds for individual model tests
            during health checks. Models that don't respond within this time
            are marked as failed. Defaults to 25.
        feature_mapping_strategy: Strategy for handling feature compatibility
            between requests and model capabilities. BEST_EFFORT attempts to
            map features even if not fully supported (e.g., simulating system
            messages), STRICT requires exact feature support and raises errors
            otherwise. Defaults to BEST_EFFORT.
        suppress_provider_errors: If True, provider errors are returned as
            error strings instead of raising exceptions. Useful for graceful
            error handling in production. Defaults to False.
        keep_raw_provider_response: Debug-only escape hatch. If True, the
            raw provider SDK response object is attached to
            ``call_record.debug.raw_provider_response`` for every successful
            call. The field is not part of ProxAI's stable contract, is not
            serialized to the query cache or ProxDash, and is mutually
            exclusive with ``cache_options`` (constructing a client with
            both raises ``ValueError``). Defaults to False.
        init_from_params: Internal parameter for initializing from a
            ProxAIClientParams object. Cannot be used together with other
            parameters.
        init_from_state: Internal parameter for restoring client state.
            Cannot be used together with other parameters.

    Raises:
        ValueError: If init_from_params or init_from_state is provided
            together with other parameters.

    Example:
        >>> import proxai as px
        >>> client = px.Client(
        ...   experiment_path="my_experiment",
        ...   cache_options=px.CacheOptions(
        ...     cache_path="/tmp/proxai_cache",
        ...     unique_response_limit=3,
        ...   ),
        ...   logging_options=px.LoggingOptions(
        ...     logging_path="/tmp/proxai_logs",
        ...     stdout=True,
        ...   ),
        ...   suppress_provider_errors=True,
        ... )
        >>> response = client.generate_text(
        ...   prompt="What is the capital of France?"
        ... )
    """
    if init_from_params is not None or init_from_state is not None:
      if (
          experiment_path is not None or cache_options is not None or
          logging_options is not None or proxdash_options is not None or
          not allow_multiprocessing or model_test_timeout != 25 or (
              feature_mapping_strategy
              != types.FeatureMappingStrategy.BEST_EFFORT
          ) or suppress_provider_errors or keep_raw_provider_response
      ):
        raise ValueError(
            "init_from_params or init_from_state cannot be set at with "
            "direct arguments. Please use one of init_from_params, "
            "init_from_state, or direct arguments.\n"
            "experiment_path: {experiment_path}\n"
            "cache_options: {cache_options}\n"
            "logging_options: {logging_options}\n"
            "proxdash_options: {proxdash_options}\n"
            "allow_multiprocessing: {allow_multiprocessing}\n"
            "model_test_timeout: {model_test_timeout}\n"
            "feature_mapping_strategy: {feature_mapping_strategy}\n"
            "suppress_provider_errors: {suppress_provider_errors}\n"
            "keep_raw_provider_response: {keep_raw_provider_response}\n"
            "init_from_params: {init_from_params}\n"
            "init_from_state: {init_from_state}\n"
        )
    else:
      init_from_params = ProxAIClientParams(
          experiment_path=experiment_path,
          cache_options=cache_options,
          logging_options=logging_options,
          proxdash_options=proxdash_options,
          allow_multiprocessing=allow_multiprocessing,
          model_test_timeout=model_test_timeout,
          feature_mapping_strategy=feature_mapping_strategy,
          suppress_provider_errors=suppress_provider_errors,
          keep_raw_provider_response=keep_raw_provider_response,
      )

    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

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
      self.keep_raw_provider_response = (
          init_from_params.keep_raw_provider_response
      )

      _ = self.model_cache_manager
      _ = self.query_cache_manager
      # BEGIN: Refactoring: Revert this after testing
      # self._init_proxdash_connection()
      # END: Refactoring


      if self.cache_options and self.cache_options.clear_model_cache_on_connect:
        self.model_cache_manager.clear_cache()
      if self.cache_options and self.cache_options.clear_query_cache_on_connect:
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
          keep_raw_provider_response=self.keep_raw_provider_response,
      )
      self._available_models_instance = available_models.AvailableModels(
          init_from_params=available_models_params
      )

    self._validate_raw_provider_response_options()
    self._maybe_emit_raw_provider_response_warning()

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _PROXAI_CLIENT_STATE_PROPERTY

  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    return types.ProxAIClientState

  def _init_default_model_cache_manager(self):
    try:
      app_dirs = platformdirs.PlatformDirs(appname="proxai", appauthor="proxai")
      self.default_model_cache_path = app_dirs.user_cache_dir
      os.makedirs(self.default_model_cache_path, exist_ok=True)
      # 4 hours cache duration makes sense for local development if proxai is
      # using platform app cache directory
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=types.CacheOptions(
              cache_path=self.default_model_cache_path,
              model_cache_duration=60 * 60 * 4,
          )
      )
      self.default_model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params
      )
      self.platform_used_for_default_model_cache = True
    except Exception:
      self.default_model_cache_path = tempfile.TemporaryDirectory()
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=types.CacheOptions(
              cache_path=self.default_model_cache_path.name
          )
      )
      self.default_model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params
      )
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
    self.registered_models = {}
    self.model_cache_manager = None
    self.query_cache_manager = None
    self.proxdash_connection = None
    self._init_default_model_cache_manager()

    self.feature_mapping_strategy = types.FeatureMappingStrategy.BEST_EFFORT
    self.suppress_provider_errors = False
    self.keep_raw_provider_response = False
    self.allow_multiprocessing = True
    self.model_test_timeout = 25

    self.available_models_instance = None

  def _validate_raw_provider_response_options(self):
    """Reject keep_raw_provider_response=True while a query cache is set.

    Runs after both the direct-kwargs and load_state branches of __init__
    converge, so it protects every construction path — including state
    restoration, which goes through
    set_property_value_without_triggering_getters and would otherwise
    bypass any validation living in the property setter.
    """
    if (self.keep_raw_provider_response and self.cache_options is not None):
      raise ValueError(
          "keep_raw_provider_response=True is incompatible with "
          "cache_options. The query cache cannot reconstruct provider "
          "SDK objects on cache hits, so the combination would silently "
          "return None for cached calls. To use both, construct two "
          "clients: a cached production client and a separate debug "
          "client with keep_raw_provider_response=True."
      )

  def _maybe_emit_raw_provider_response_warning(self):
    if not self.keep_raw_provider_response:
      return
    logging_utils.log_message(
        logging_options=self.logging_options,
        message=(
            "keep_raw_provider_response=True is a debugging-only escape "
            "hatch. The raw provider response is not part of ProxAI's "
            "stable contract, is not serialized to the query cache or "
            "ProxDash, and may break at any provider SDK upgrade. It is "
            "also mutually exclusive with cache_options. If you need a "
            "specific provider field surfaced as a first-class CallRecord "
            "attribute, please reach out to the ProxAI team so we can "
            "model it properly instead of having you depend on this hatch "
            "long-term."
        ),
        type=types.LoggingType.WARNING,
    )

  @property
  def run_type(self) -> types.RunType:
    return self.get_property_value("run_type")

  @run_type.setter
  def run_type(self, value: types.RunType):
    self.set_property_value("run_type", value)

  @property
  def hidden_run_key(self) -> str:
    return self.get_property_value("hidden_run_key")

  @hidden_run_key.setter
  def hidden_run_key(self, value: str):
    self.set_property_value("hidden_run_key", value)

  @property
  def experiment_path(self) -> str | None:
    return self.get_property_value("experiment_path")

  @experiment_path.setter
  def experiment_path(self, value: str | None):
    if value is not None:
      experiment.validate_experiment_path(value)
    self.set_property_value("experiment_path", value)

  @property
  def default_model_cache_path(self) -> str | None:
    return self.get_property_value("default_model_cache_path")

  @default_model_cache_path.setter
  def default_model_cache_path(self, value: str | None):
    self.set_property_value("default_model_cache_path", value)

  @property
  def root_logging_path(self) -> str | None:
    return self.get_property_value("root_logging_path")

  @root_logging_path.setter
  def root_logging_path(self, value: str | None):
    if value and not os.path.exists(value):
      raise ValueError(f"Root logging path does not exist: {value}")
    self.set_property_value("root_logging_path", value)

  @property
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value("logging_options")

  @logging_options.setter
  def logging_options(self, value: types.LoggingOptions | None):
    root_logging_path = None
    if value and value.logging_path:
      root_logging_path = value.logging_path
    else:
      root_logging_path = None

    result_logging_options = types.LoggingOptions()
    if root_logging_path is not None:
      if self.experiment_path is not None:
        result_logging_options.logging_path = os.path.join(
            root_logging_path, self.experiment_path
        )
      else:
        result_logging_options.logging_path = root_logging_path
      if not os.path.exists(result_logging_options.logging_path):
        os.makedirs(result_logging_options.logging_path, exist_ok=True)
    else:
      result_logging_options.logging_path = None

    if value is not None:
      result_logging_options.stdout = value.stdout
      result_logging_options.hide_sensitive_content = (
          value.hide_sensitive_content
      )

    self.set_property_value("logging_options", result_logging_options)

  @property
  def cache_options(self) -> types.CacheOptions:
    return self.get_property_value("cache_options")

  @cache_options.setter
  def cache_options(self, value: types.CacheOptions | None):
    result_cache_options = None
    if value is not None:
      if not value.cache_path and not value.disable_model_cache:
        raise ValueError("cache_path is required while setting cache_options")
      result_cache_options = types.CacheOptions()

      result_cache_options.cache_path = value.cache_path

      result_cache_options.unique_response_limit = value.unique_response_limit
      result_cache_options.retry_if_error_cached = value.retry_if_error_cached
      result_cache_options.clear_query_cache_on_connect = (
          value.clear_query_cache_on_connect
      )

      result_cache_options.disable_model_cache = value.disable_model_cache
      result_cache_options.clear_model_cache_on_connect = (
          value.clear_model_cache_on_connect
      )
      result_cache_options.model_cache_duration = value.model_cache_duration

    self.set_property_value("cache_options", result_cache_options)

  @property
  def proxdash_options(self) -> types.ProxDashOptions:
    return self.get_property_value("proxdash_options")

  @proxdash_options.setter
  def proxdash_options(self, value: types.ProxDashOptions | None):
    result_proxdash_options = types.ProxDashOptions()
    if value is not None:
      result_proxdash_options.stdout = value.stdout
      result_proxdash_options.hide_sensitive_content = (
          value.hide_sensitive_content
      )
      result_proxdash_options.disable_proxdash = value.disable_proxdash
      result_proxdash_options.api_key = value.api_key
      result_proxdash_options.base_url = value.base_url

    self.set_property_value("proxdash_options", result_proxdash_options)

  @property
  def model_configs_requested_from_proxdash(self) -> bool:
    return self.get_property_value("model_configs_requested_from_proxdash")

  @model_configs_requested_from_proxdash.setter
  def model_configs_requested_from_proxdash(self, value: bool):
    self.set_property_value("model_configs_requested_from_proxdash", value)

  @property
  def allow_multiprocessing(self) -> bool:
    return self.get_property_value("allow_multiprocessing")

  @allow_multiprocessing.setter
  def allow_multiprocessing(self, value: bool):
    self.set_property_value("allow_multiprocessing", value)

  @property
  def model_test_timeout(self) -> int:
    return self.get_property_value("model_test_timeout")

  @model_test_timeout.setter
  def model_test_timeout(self, value: int):
    if value < 1:
      raise ValueError("model_test_timeout must be greater than 0.")
    self.set_property_value("model_test_timeout", value)

  @property
  def feature_mapping_strategy(self) -> types.FeatureMappingStrategy:
    return self.get_property_value("feature_mapping_strategy")

  @feature_mapping_strategy.setter
  def feature_mapping_strategy(self, value: types.FeatureMappingStrategy):
    self.set_property_value("feature_mapping_strategy", value)

  @property
  def suppress_provider_errors(self) -> bool:
    return self.get_property_value("suppress_provider_errors")

  @suppress_provider_errors.setter
  def suppress_provider_errors(self, value: bool):
    self.set_property_value("suppress_provider_errors", value)

  @property
  def keep_raw_provider_response(self) -> bool:
    return self.get_property_value("keep_raw_provider_response")

  @keep_raw_provider_response.setter
  def keep_raw_provider_response(self, value: bool):
    # Validation lives in _validate_raw_provider_response_options() so that
    # it runs on every construction path, including the load_state path
    # which bypasses property setters.
    self.set_property_value("keep_raw_provider_response", value)

  @property
  def model_configs_instance(self) -> model_configs.ModelConfigs:
    if (
        not self.model_configs_requested_from_proxdash and
        self.proxdash_connection
    ):
      model_configs_schema = self.proxdash_connection.get_model_configs_schema()
      if model_configs_schema is not None:
        self._model_configs_instance.model_configs_schema = model_configs_schema
      self.model_configs_requested_from_proxdash = True
    return self.get_property_value("model_configs_instance")

  @model_configs_instance.setter
  def model_configs_instance(self, value: model_configs.ModelConfigs):
    self.set_property_value("model_configs_instance", value)

  @property
  def default_model_cache_manager(self) -> model_cache.ModelCacheManager:
    return self.get_property_value("default_model_cache_manager")

  @default_model_cache_manager.setter
  def default_model_cache_manager(self, value: model_cache.ModelCacheManager):
    self.set_property_value("default_model_cache_manager", value)

  @property
  def model_cache_manager(self) -> model_cache.ModelCacheManager:
    if self.cache_options is None:
      return self.default_model_cache_manager

    if self._model_cache_manager is None:
      model_cache_manager_params = model_cache.ModelCacheManagerParams(
          cache_options=self.cache_options
      )
      self._model_cache_manager = model_cache.ModelCacheManager(
          init_from_params=model_cache_manager_params
      )
    return self.get_property_value("model_cache_manager")

  @model_cache_manager.setter
  def model_cache_manager(self, value: model_cache.ModelCacheManager):
    self.set_property_value("model_cache_manager", value)

  @property
  def query_cache_manager(self) -> query_cache.QueryCacheManager:
    if self._query_cache_manager is None and self.cache_options is not None:
      query_cache_manager_params = query_cache.QueryCacheManagerParams(
          cache_options=self.cache_options
      )
      self._query_cache_manager = query_cache.QueryCacheManager(
          init_from_params=query_cache_manager_params
      )
    return self.get_property_value("query_cache_manager")

  @query_cache_manager.setter
  def query_cache_manager(self, value: query_cache.QueryCacheManager):
    self.set_property_value("query_cache_manager", value)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_property_value("proxdash_connection")

  @proxdash_connection.setter
  def proxdash_connection(self, value: proxdash.ProxDashConnection):
    self.set_property_value("proxdash_connection", value)

  @property
  def available_models_instance(self) -> available_models.AvailableModels:
    return self.get_state_controlled_property_value("available_models_instance")

  @available_models_instance.setter
  def available_models_instance(self, value: available_models.AvailableModels):
    self.set_state_controlled_property_value("available_models_instance", value)

  @property
  def models(self) -> ModelConnector:
    """Access model discovery and availability information.

    Provides an interface for querying available models, providers, and
    checking which models are currently working. This property returns a
    ModelConnector instance bound to this client.

    Returns:
        ModelConnector: Interface for querying available models.

    Example:
        >>> client = px.Client()
        >>> # List all available models
        >>> all_models = client.models.list_models()
        >>> # List working models only
        >>> working = client.models.list_working_models(verbose=False)
        >>> # Get a specific model
        >>> model = client.models.get_model("openai", "gpt-4")
    """
    if not hasattr(self, "_models_connector") or self._models_connector is None:
      self._models_connector = ModelConnector(lambda: self)
    return self._models_connector

  def _init_proxdash_connection(self):
    if self._proxdash_connection is None:
      proxdash_connection_params = proxdash.ProxDashConnectionParams(
          hidden_run_key=self.hidden_run_key,
          experiment_path=self.experiment_path,
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
      )
      self._proxdash_connection = proxdash.ProxDashConnection(
          init_from_params=proxdash_connection_params
      )

  def get_default_provider_model(
      self, call_type: types.CallType,
  ) -> types.ProviderModelType:
    """Resolve and return the default ProviderModelType for a call type."""
    if call_type != types.CallType.TEXT:
      raise ValueError(f"Call type not supported: {call_type}")

    if call_type in self.registered_models:
      return self.registered_models[call_type]

    default_models = (
        self.model_configs_instance.get_default_model_priority_list()
    )
    for provider_model in default_models:
      try:
        self.available_models_instance.get_working_model(
            provider=provider_model.provider, model=provider_model.model
        )
        self.registered_models[call_type] = provider_model
        self.registered_model_connectors[call_type] = (
            self.available_models_instance.
            get_model_connector(provider_model)
        )
        return provider_model
      except ValueError:
        continue

    models = self.available_models_instance.list_working_models()
    if len(models.working_models) > 0:
      provider_model = models.working_models.pop()
      self.registered_models[call_type] = provider_model
      self.registered_model_connectors[call_type] = (
          self.available_models_instance.get_model_connector(provider_model)
      )
      return provider_model

    raise ValueError(
        "No working models found in current environment:\n"
        "* Please check your environment variables and try again.\n"
        "* You can use px.check_health() method as instructed in "
        "https://www.proxai.co/proxai-docs/check-health"
    )

  def get_registered_model_connector(self, call_type: types.CallType):
    """Get or create a connector for the default model of a call type."""
    self.get_default_provider_model(call_type=call_type)
    return self.registered_model_connectors[call_type]

  def set_model(
      self,
      provider_model: types.ProviderModelIdentifierType | None = None,
      generate_text: types.ProviderModelIdentifierType | None = None,
  ) -> None:
    """Sets the default model for text generation requests.

    Configures which AI provider and model should be used for subsequent
    generate_text() calls when no explicit provider_model is specified.

    Args:
        provider_model: The provider and model to use as default. Can be
            specified as a tuple like ('openai', 'gpt-4') or a
            ProviderModelType instance.
        generate_text: Alias for provider_model. Use this parameter name
            for clarity when setting the model specifically for text
            generation. Cannot be used together with provider_model.

    Raises:
        ValueError: If both provider_model and generate_text are provided,
            or if neither is provided.

    Example:
        >>> client = px.Client()
        >>> client.set_model(provider_model=("openai", "gpt-4"))
        >>> # Or equivalently:
        >>> client.set_model(generate_text=("anthropic", "claude-3-opus"))
    """
    if provider_model and generate_text:
      raise ValueError(
          "provider_model and generate_text cannot be set at the "
          "same time. Please set one of them."
      )

    if provider_model is None and generate_text is None:
      raise ValueError("provider_model or generate_text must be set.")

    if generate_text:
      provider_model = generate_text

    self.model_configs_instance.check_provider_model_identifier_type(
        provider_model
    )

    resolved = self.model_configs_instance.get_provider_model(provider_model)
    self.registered_models[types.CallType.TEXT] = resolved
    self.registered_model_connectors[
        types.CallType.TEXT
    ] = (self.available_models_instance.get_model_connector(provider_model))

  def generate(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      response_format: types.ResponseFormatParam | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> types.CallRecord:
    if prompt is not None and messages is not None:
      raise ValueError('prompt and messages cannot be used together')

    if system_prompt is not None and messages is not None:
      raise ValueError(
          'system_prompt and messages cannot be used together. '
          'Please use "system" message in messages to set the system prompt.\n'
          'px.generate(\n'
          '    messages=[\n'
          '        {"role": "system",\n'
          '         "content": "You are a helpful assistant."},\n'
          '        ...])')

    if (connection_options and
        connection_options.fallback_models and
        connection_options.suppress_provider_errors):
      raise ValueError(
          'suppress_provider_errors and fallback_models cannot be '
          'used together.\n'
          f'connection_options: {connection_options}')

    if (connection_options and
        connection_options.endpoint and
        connection_options.fallback_models):
      raise ValueError(
          'endpoint and fallback_models cannot be used together.\n'
          f'connection_options: {connection_options}')

    if connection_options is None:
      connection_options = types.ConnectionOptions()
    if connection_options.suppress_provider_errors is None:
      connection_options.suppress_provider_errors = (
          self.suppress_provider_errors)

    if (connection_options.override_cache_value and (
        self.query_cache_manager is None or
        self.query_cache_manager.status
        != types.QueryCacheManagerStatus.WORKING)):
      raise ValueError(
          "override_cache_value is True but query cache is not configured.\n"
          "Please set cache_options to enable query cache.")

    messages = type_utils.messages_param_to_chat(messages)
    response_format = type_utils.response_format_param_to_response_format(
        response_format)

    provider_models = [
        self.model_configs_instance.get_provider_model(provider_model)]

    if connection_options.fallback_models:
      for fallback_model in connection_options.fallback_models:
        provider_models.append(
            self.model_configs_instance.get_provider_model(fallback_model))
      connection_options.suppress_provider_errors = True
      connection_options.fallback_models = None

    connection_metadata = types.ConnectionMetadata(
        feature_mapping_strategy=self.feature_mapping_strategy)
    for idx, provider_model in enumerate(provider_models):
      model_connector = self.available_models_instance.get_model_connector(
          provider_model_identifier=provider_model
      )
      provider_model_config = (
          self.model_configs_instance.get_provider_model_config(provider_model))
      result_record = model_connector.generate(
          prompt=prompt,
          messages=messages,
          system_prompt=system_prompt,
          provider_model=provider_model,
          provider_model_config=provider_model_config,
          parameters=parameters,
          tools=tools,
          response_format=response_format,
          connection_options=connection_options,
          connection_metadata=connection_metadata,
      )
      if result_record.result.status == types.ResultStatusType.SUCCESS:
        return result_record
      if idx == 0:
        connection_metadata.failed_fallback_models = []
      connection_metadata.failed_fallback_models.append(provider_model)
    return result_record

  def generate_text(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> str:
    """Generates text using the configured AI model.

    Thin alias for generate() that resolves the default model and returns
    the generated text string directly.

    Args:
        prompt: Simple text prompt for the AI model. Cannot be used together
            with messages parameter.
        messages: Structured messages for multi-turn conversations. Cannot
            be used together with prompt parameter.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request,
            overriding the default model.
        parameters: Generation parameters (temperature, max_tokens, etc.).
        tools: Tools to enable for this request (e.g., web search).
        connection_options: Connection options (fallback models, cache
            control, error suppression, etc.).

    Returns:
        The generated text as a string. If the provider returns an error
        and suppress_provider_errors is True, returns the error message
        string.

    Example:
        >>> client = px.Client()
        >>> response = client.generate_text(
        ...   prompt="What is the capital of France?"
        ... )
        >>> print(response)
        'The capital of France is Paris.'
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          call_type=types.CallType.TEXT)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_text

  def generate_json(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> dict:
    """Generates a JSON response using the configured AI model.

    Thin alias for generate() that resolves the default model, sets
    response_format to JSON, and returns the parsed dict directly.

    Args:
        prompt: Simple text prompt for the AI model. Cannot be used together
            with messages parameter.
        messages: Structured messages for multi-turn conversations.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request.
        parameters: Generation parameters (temperature, max_tokens, etc.).
        tools: Tools to enable for this request.
        connection_options: Connection options.

    Returns:
        The generated response as a parsed dict. If the provider returns
        an error and suppress_provider_errors is True, returns the error
        message string.

    Example:
        >>> client = px.Client()
        >>> result = client.generate_json(
        ...   prompt="Return the capital of France as JSON"
        ... )
        >>> print(result)
        {'capital': 'Paris', 'country': 'France'}
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          call_type=types.CallType.TEXT)

    response_format = types.ResponseFormat(
        type=types.ResponseFormatType.JSON)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        response_format=response_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_json

  def generate_pydantic(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      response_format: types.ResponseFormatParam | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> pydantic.BaseModel:
    """Generates a structured pydantic response using the configured AI model.

    Thin alias for generate() that resolves the default model and returns
    the pydantic model instance directly.

    Args:
        prompt: Simple text prompt for the AI model. Cannot be used together
            with messages parameter.
        messages: Structured messages for multi-turn conversations.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request.
        parameters: Generation parameters (temperature, max_tokens, etc.).
        tools: Tools to enable for this request.
        response_format: The pydantic model class to validate against.
        connection_options: Connection options.

    Returns:
        An instance of the pydantic model specified in response_format.
        If the provider returns an error and suppress_provider_errors is
        True, returns the error message string.

    Example:
        >>> from pydantic import BaseModel
        >>> class City(BaseModel):
        ...   name: str
        ...   country: str
        >>> client = px.Client()
        >>> result = client.generate_pydantic(
        ...   prompt="What is the capital of France?",
        ...   response_format=City
        ... )
        >>> print(result.name)
        'Paris'
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          call_type=types.CallType.TEXT)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        response_format=response_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_pydantic

  def get_current_options(
      self,
      json: bool = False,
  ) -> types.RunOptions | dict:
    """Returns the current configuration options of this client.

    Retrieves all active configuration settings including run type, experiment
    path, logging options, cache options, and other runtime parameters.

    Args:
        json: If True, returns the options as a JSON-serializable dictionary.
            If False, returns a RunOptions dataclass. Defaults to False.

    Returns:
        Union[types.RunOptions, dict]: The current configuration as either
            a RunOptions dataclass or a dictionary depending on the json
            parameter.

    Example:
        >>> client = px.Client(
        ...   cache_options=px.CacheOptions(cache_path="/tmp/cache")
        ... )
        >>> options = client.get_current_options()
        >>> print(options.cache_options.cache_path)
        '/tmp/cache'

        >>> # Get as JSON-serializable dict
        >>> options_dict = client.get_current_options(json=True)
    """
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
        keep_raw_provider_response=self.keep_raw_provider_response,
        allow_multiprocessing=self.allow_multiprocessing,
        model_test_timeout=self.model_test_timeout,
    )
    if json:
      return type_serializer.encode_run_options(run_options=run_options)
    return run_options

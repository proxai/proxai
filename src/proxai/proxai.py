
import proxai.client as client
import proxai.types as types

# Re-export for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions
ResponseFormat = types.StructuredResponseFormat
ResponseFormatType = types.ResponseFormatType

_DEFAULT_CLIENT: client.ProxAIClient | None = None
Client = client.ProxAIClient


def get_default_proxai_client() -> client.ProxAIClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        proxai_client_params = client.ProxAIClientParams()
        _DEFAULT_CLIENT = client.ProxAIClient(
            init_from_params=proxai_client_params)
    return _DEFAULT_CLIENT


def connect(
    experiment_path: str | None = None,
    cache_options: types.CacheOptions | None = None,
    logging_options: types.LoggingOptions | None = None,
    proxdash_options: types.ProxDashOptions | None = None,
    allow_multiprocessing: bool | None = True,
    model_test_timeout: int | None = 25,
    feature_mapping_strategy: (
        types.FeatureMappingStrategy | None
    ) = types.FeatureMappingStrategy.BEST_EFFORT,
    suppress_provider_errors: bool | None = False,
) -> None:
  """Initializes the default ProxAI client with the specified configuration.

  This function sets up the global ProxAI client that will be used for all
  subsequent API calls. It configures caching, logging, ProxDash integration,
  and various behavior options.

  Args:
      experiment_path: Path identifier for organizing experiments. Used to
          group related API calls and logs together.
      cache_options: Configuration for query and model caching behavior.
          Controls cache paths, response limits, and cache clearing options.
      logging_options: Configuration for logging behavior. Controls log file
          paths, stdout output, and sensitive content handling.
      proxdash_options: Configuration for ProxDash monitoring integration.
          Controls API key, base URL, and output options.
      allow_multiprocessing: Whether to test models in parallel using
          multiprocessing. Disable if you encounter process spawning errors
          (common in Jupyter notebooks, AWS Lambda, or on Windows/macOS
          without proper multiprocessing guards). Defaults to True.
      model_test_timeout: Timeout in seconds for individual model tests
          during health checks. Defaults to 25.
      feature_mapping_strategy: Strategy for handling feature compatibility
          between requests and model capabilities. BEST_EFFORT attempts to
          map features even if not fully supported, STRICT requires exact
          matches. Defaults to BEST_EFFORT.
      suppress_provider_errors: If True, provider errors are returned as
          strings instead of raising exceptions. Defaults to False.

  Returns:
      None

  Example:
      >>> import proxai as px
      >>> px.connect(
      ...     cache_options=px.CacheOptions(cache_path='/tmp/cache'),
      ...     logging_options=px.LoggingOptions(stdout=True),
      ... )
  """
  global _DEFAULT_CLIENT
  proxai_client_params = client.ProxAIClientParams(
      experiment_path=experiment_path,
      cache_options=cache_options,
      logging_options=logging_options,
      proxdash_options=proxdash_options,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
      feature_mapping_strategy=feature_mapping_strategy,
      suppress_provider_errors=suppress_provider_errors,
  )
  _DEFAULT_CLIENT = client.ProxAIClient(init_from_params=proxai_client_params)


def generate_text(
    prompt: str | None = None,
    system: str | None = None,
    messages: types.MessagesType | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop: types.StopType | None = None,
    response_format: types.ResponseFormatParam | None = None,
    web_search: bool | None = None,
    provider_model: types.ProviderModelIdentifierType | None = None,
    feature_mapping_strategy: types.FeatureMappingStrategy | None = None,
    use_cache: bool | None = None,
    unique_response_limit: int | None = None,
    extensive_return: bool = False,
    suppress_provider_errors: bool | None = None,
) -> str | types.LoggingRecord:
  """Generates text using the configured AI model.

  Sends a text generation request to the AI provider using either a simple
  prompt or a structured messages format. Supports various configuration
  options including response formatting, caching, and error handling.

  Args:
      prompt: Simple text prompt for the AI model. Cannot be used together
          with messages parameter.
      system: System message to set the AI's behavior and context. Provides
          instructions that guide the model's responses.
      messages: List of message dictionaries for multi-turn conversations.
          Each message should have 'role' and 'content' keys. Cannot be
          used together with prompt parameter.
      max_tokens: Maximum number of tokens to generate in the response.
          If not specified, uses the model's default limit.
      temperature: Sampling temperature between 0 and 2. Higher values
          (e.g., 0.8) make output more random, lower values (e.g., 0.2)
          make it more deterministic.
      stop: String or list of strings that will stop generation when
          encountered in the output.
      response_format: Specifies the desired response format. Can be a
          Pydantic model class for structured output, a JSON schema dict,
          or a StructuredResponseFormat instance.
      web_search: Whether to enable web search capabilities for models
          that support it.
      provider_model: Specific provider and model to use for this request,
          overriding the default model. Can be a tuple like
          ('openai', 'gpt-4') or a ProviderModelType instance.
      feature_mapping_strategy: Strategy for handling feature compatibility.
          Overrides the client-level setting for this request.
      use_cache: Whether to use query caching for this request. If None,
          uses cache if available and configured.
      unique_response_limit: Number of unique responses to collect before
          returning from cache. Useful for generating diverse outputs.
      extensive_return: If True, returns the full LoggingRecord with
          metadata instead of just the response text.
      suppress_provider_errors: If True, returns error messages as strings
          instead of raising exceptions. Overrides client-level setting.

  Returns:
      Union[str, types.LoggingRecord]: The generated text response as a
          string, or the full LoggingRecord if extensive_return is True.
          If response_format specifies a Pydantic model, returns an
          instance of that model.

  Raises:
      ValueError: If both prompt and messages are provided, or if use_cache
          is True but cache is not configured.
      Exception: If the provider returns an error and suppress_provider_errors
          is False.

  Example:
      >>> import proxai as px
      >>> response = px.generate_text(prompt='What is the capital of France?')
      >>> print(response)
      'The capital of France is Paris.'

      >>> # Using structured output
      >>> from pydantic import BaseModel
      >>> class City(BaseModel):
      ...     name: str
      ...     country: str
      >>> result = px.generate_text(
      ...     prompt='What is the capital of France?',
      ...     response_format=City
      ... )
      >>> print(result.name)
      'Paris'
  """
  return get_default_proxai_client().generate_text(
      prompt=prompt,
      system=system,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop,
      response_format=response_format,
      web_search=web_search,
      provider_model=provider_model,
      feature_mapping_strategy=feature_mapping_strategy,
      use_cache=use_cache,
      unique_response_limit=unique_response_limit,
      extensive_return=extensive_return,
      suppress_provider_errors=suppress_provider_errors,
  )


def set_model(
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

  Returns:
      None

  Raises:
      ValueError: If both provider_model and generate_text are provided,
          or if neither is provided.

  Example:
      >>> import proxai as px
      >>> px.set_model(provider_model=('openai', 'gpt-4'))
      >>> # Or equivalently:
      >>> px.set_model(generate_text=('anthropic', 'claude-3-opus'))
  """
  get_default_proxai_client().set_model(
      provider_model=provider_model,
      generate_text=generate_text)


def check_health(
    experiment_path: str | None = None,
    verbose: bool = True,
    allow_multiprocessing: bool = True,
    model_test_timeout: int = 25,
    extensive_return: bool = False,
) -> types.ModelStatus | None:
  """Tests connectivity and response times for all available AI models.

  Performs a health check by sending test requests to each configured AI
  model and reporting which models are working and which have failed.
  Results are cached to speed up subsequent checks.

  Args:
      experiment_path: Path identifier for organizing the health check
          experiment logs.
      verbose: If True, prints detailed progress and results to stdout
          including per-model response times. Defaults to True.
      allow_multiprocessing: Whether to test models in parallel using
          multiprocessing. Disable if you encounter process spawning errors
          (common in Jupyter notebooks, AWS Lambda, or on Windows/macOS
          without proper multiprocessing guards). Defaults to True.
      model_test_timeout: Maximum time in seconds to wait for each model
          to respond before marking it as failed. Defaults to 25.
      extensive_return: If True, returns the full ModelStatus object
          containing detailed information about all tested models.
          Defaults to False.

  Returns:
      Optional[types.ModelStatus]: If extensive_return is True, returns
          a ModelStatus object containing working_models, failed_models,
          and provider_queries with detailed logging records. Returns
          None if extensive_return is False.

  Example:
      >>> import proxai as px
      >>> # Quick health check with console output
      >>> px.check_health()
      > Starting to test each model...
      > Finished testing.
         Registered Providers: 3
         Succeeded Models: 5
         Failed Models: 1
      > anthropic:
         [ WORKING |   1.23s ]: claude-3-opus
         ...

      >>> # Get detailed results programmatically
      >>> status = px.check_health(verbose=False, extensive_return=True)
      >>> print(len(status.working_models))
      5
  """
  px_client_params = client.ProxAIClientParams(
      experiment_path=experiment_path,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
  )
  px_client = client.ProxAIClient(init_from_params=px_client_params)
  if verbose:
    print('> Starting to test each model...')
  model_status = px_client.available_models_instance.list_working_models(
      clear_model_cache=True,
      verbose=verbose,
      return_all=True)
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
        provider_model = px_client.model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ WORKING | {duration.total_seconds():6.2f}s ]: {model}')
      for model in sorted(result_table[provider]['failed']):
        provider_model = px_client.model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ FAILED  | {duration.total_seconds():6.2f}s ]: {model}')
  if extensive_return:
    return model_status


def get_current_options(
    json: bool = False,
) -> types.RunOptions | dict:
  """Returns the current configuration options of the default ProxAI client.

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
      >>> import proxai as px
      >>> px.connect(cache_options=px.CacheOptions(cache_path='/tmp/cache'))
      >>> options = px.get_current_options()
      >>> print(options.cache_options.cache_path)
      '/tmp/cache'

      >>> # Get as JSON-serializable dict
      >>> options_dict = px.get_current_options(json=True)
  """
  return get_default_proxai_client().get_current_options(json=json)


def reset_state() -> None:
  """Resets the global ProxAI client state and clears cached data.

  Clears the model cache if it was created using platform directories,
  then resets the default client to None. After calling this function,
  a new client will be created on the next API call or connect() call.

  This is useful for:
  - Testing scenarios where a fresh client state is needed
  - Clearing cached model availability data
  - Releasing resources held by the client

  Returns:
      None

  Example:
      >>> import proxai as px
      >>> px.connect(cache_options=px.CacheOptions(cache_path='/tmp/cache'))
      >>> px.generate_text(prompt='Hello')
      >>> px.reset_state()  # Clear state and cache
      >>> # Next call will create a new client
  """
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    return
  if _DEFAULT_CLIENT.platform_used_for_default_model_cache:
    _DEFAULT_CLIENT.model_cache_manager.clear_cache()
  _DEFAULT_CLIENT = None


class DefaultModelsConnector:
  """Provides access to model discovery and availability information.

  This class offers methods to list available models, providers, and check
  which models are currently working. It is typically accessed via the
  ``px.models`` singleton.

  Example:
      >>> import proxai as px
      >>> # List all available models
      >>> models = px.models.list_models()
      >>> # List working models only
      >>> working = px.models.list_working_models()
  """

  def list_models(
      self,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureListParam | None = None,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """Lists all configured models matching the specified criteria.

    Returns models that are configured in the system regardless of whether
    they are currently accessible or working. Use list_working_models()
    to get only models that have been verified to work.

    Args:
        model_size: Filter by model size category. Can be a ModelSizeType
            enum value ('small', 'medium', 'large', 'largest') or a string.
        features: Filter by required features. List of feature names that
            models must support (e.g., ['system', 'temperature']).
        call_type: The type of API call to filter models for.
            Defaults to GENERATE_TEXT.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            ProviderModelType objects, or a ModelStatus object if return_all
            is True.

    Example:
        >>> import proxai as px
        >>> # List all models
        >>> models = px.models.list_models()
        >>> print(models[0])
        (openai, gpt-4)

        >>> # Filter by size
        >>> large_models = px.models.list_models(model_size='large')
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_models(
            model_size=model_size,
            features=features,
            call_type=call_type,
        ))

  def list_providers(
      self,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
  ) -> list[str]:
    """Lists all providers that have API keys configured.

    Returns provider names for which the required environment variables
    are set, indicating the provider can potentially be used.

    Args:
        call_type: The type of API call to filter providers for.
            Defaults to GENERATE_TEXT.

    Returns:
        List[str]: A sorted list of provider names (e.g., ['anthropic',
            'openai']).

    Example:
        >>> import proxai as px
        >>> providers = px.models.list_providers()
        >>> print(providers)
        ['anthropic', 'google', 'openai']
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_providers(
            call_type=call_type,
        ))

  def list_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureListParam | None = None,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
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
        call_type: The type of API call to filter models for.
            Defaults to GENERATE_TEXT.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            ProviderModelType objects, or a ModelStatus object if return_all
            is True.

    Raises:
        ValueError: If the provider's API key is not found in environment
            variables.

    Example:
        >>> import proxai as px
        >>> openai_models = px.models.list_provider_models('openai')
        >>> print(openai_models)
        [(openai, gpt-4), (openai, gpt-3.5-turbo), ...]
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_provider_models(
            provider=provider,
            model_size=model_size,
            features=features,
            call_type=call_type,
        ))

  def get_model(
      self,
      provider: str,
      model: str,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
  ) -> types.ProviderModelType:
    """Gets a specific model by provider and model name.

    Returns the ProviderModelType for the specified provider and model
    combination if it exists and the provider's API key is configured.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model: The model name (e.g., 'gpt-4', 'claude-3-opus').
        call_type: The type of API call the model should support.
            Defaults to GENERATE_TEXT.

    Returns:
        types.ProviderModelType: The model information including provider,
            model name, and provider-specific identifier.

    Raises:
        ValueError: If the provider's API key is not found, or if the
            model doesn't exist or doesn't support the specified call_type.

    Example:
        >>> import proxai as px
        >>> model = px.models.get_model('openai', 'gpt-4')
        >>> print(model)
        (openai, gpt-4)
    """
    return (
        get_default_proxai_client()
        .available_models_instance.get_model(
            provider=provider,
            model=model,
            call_type=call_type,
        ))

  def list_working_models(
      self,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureListParam | None = None,
      verbose: bool = True,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
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
        call_type: The type of API call to test. Defaults to GENERATE_TEXT.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            working ProviderModelType objects, or a ModelStatus object if
            return_all is True.

    Example:
        >>> import proxai as px
        >>> working_models = px.models.list_working_models(verbose=False)
        >>> print(f'Found {len(working_models)} working models')
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_models(
            model_size=model_size,
            features=features,
            verbose=verbose,
            return_all=return_all,
            clear_model_cache=clear_model_cache,
            call_type=call_type,
        ))

  def list_working_providers(
      self,
      verbose: bool = True,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
  ) -> list[str]:
    """Lists providers that have at least one working model.

    Tests models and returns providers that have successfully responded
    to at least one test request.

    Args:
        verbose: If True, prints progress information during testing.
            Defaults to True.
        clear_model_cache: If True, clears the model cache and retests
            all models. Defaults to False.
        call_type: The type of API call to test. Defaults to GENERATE_TEXT.

    Returns:
        List[str]: A sorted list of provider names with working models.

    Example:
        >>> import proxai as px
        >>> providers = px.models.list_working_providers(verbose=False)
        >>> print(providers)
        ['anthropic', 'openai']
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_providers(
            verbose=verbose,
            clear_model_cache=clear_model_cache,
            call_type=call_type,
        ))

  def list_working_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureListParam | None = None,
      verbose: bool = True,
      return_all: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
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
        call_type: The type of API call to test. Defaults to GENERATE_TEXT.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A list of
            working ProviderModelType objects from the provider, or a
            ModelStatus object if return_all is True.

    Raises:
        ValueError: If the provider's API key is not found.

    Example:
        >>> import proxai as px
        >>> openai_working = px.models.list_working_provider_models(
        ...     'openai', verbose=False)
        >>> print(openai_working)
        [(openai, gpt-4), (openai, gpt-3.5-turbo)]
    """
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_provider_models(
            provider=provider,
            model_size=model_size,
            features=features,
            verbose=verbose,
            return_all=return_all,
            clear_model_cache=clear_model_cache,
            call_type=call_type,
        ))

  def get_working_model(
      self,
      provider: str,
      model: str,
      verbose: bool = False,
      clear_model_cache: bool = False,
      call_type: types.CallType = types.CallType.GENERATE_TEXT,
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
        call_type: The type of API call to test. Defaults to GENERATE_TEXT.

    Returns:
        types.ProviderModelType: The model information if the model is
            working.

    Raises:
        ValueError: If the provider's API key is not found, the model
            doesn't exist, or the model failed the health check.

    Example:
        >>> import proxai as px
        >>> model = px.models.get_working_model('openai', 'gpt-4')
        >>> print(model)
        (openai, gpt-4)
    """
    return (
        get_default_proxai_client()
        .available_models_instance.get_working_model(
            provider=provider,
            model=model,
            verbose=verbose,
            clear_model_cache=clear_model_cache,
            call_type=call_type,
        ))

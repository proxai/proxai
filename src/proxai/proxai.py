import proxai.client as client
import proxai.types as types
from proxai.client import ModelConnector

# Re-export for easy access
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions
ResponseFormat = types.StructuredResponseFormat
ResponseFormatType = types.ResponseFormatType
FeatureMappingStrategy = types.FeatureMappingStrategy
ProviderModelType = types.ProviderModelType

_DEFAULT_CLIENT: client.ProxAIClient | None = None
Client = client.ProxAIClient


def get_default_proxai_client() -> client.ProxAIClient:
  """Return the global default ProxAI client, creating it if needed."""
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    proxai_client_params = client.ProxAIClientParams()
    _DEFAULT_CLIENT = client.ProxAIClient(init_from_params=proxai_client_params)
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
      ...   cache_options=px.CacheOptions(cache_path="/tmp/cache"),
      ...   logging_options=px.LoggingOptions(stdout=True),
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
      >>> response = px.generate_text(prompt="What is the capital of France?")
      >>> print(response)
      'The capital of France is Paris.'

      >>> # Using structured output
      >>> from pydantic import BaseModel
      >>> class City(BaseModel):
      ...   name: str
      ...   country: str
      >>> result = px.generate_text(
      ...   prompt="What is the capital of France?", response_format=City
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
      >>> px.set_model(provider_model=("openai", "gpt-4"))
      >>> # Or equivalently:
      >>> px.set_model(generate_text=("anthropic", "claude-3-opus"))
  """
  get_default_proxai_client().set_model(
    provider_model=provider_model, generate_text=generate_text
  )


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
    print("> Starting to test each model...")
  model_status = px_client.available_models_instance.list_working_models(
    clear_model_cache=True, verbose=verbose, return_all=True
  )
  if verbose:
    providers = set(
      [model.provider for model in model_status.working_models]
      + [model.provider for model in model_status.failed_models]
    )
    result_table = {
      provider: {"working": [], "failed": []} for provider in providers
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
        provider_model = px_client.model_configs_instance.get_provider_model(
          (provider, model)
        )
        duration = model_status.provider_queries[
          provider_model
        ].response_record.response_time
        print(f"   [ WORKING | {duration.total_seconds():6.2f}s ]: {model}")
      for model in sorted(result_table[provider]["failed"]):
        provider_model = px_client.model_configs_instance.get_provider_model(
          (provider, model)
        )
        duration = model_status.provider_queries[
          provider_model
        ].response_record.response_time
        print(f"   [ FAILED  | {duration.total_seconds():6.2f}s ]: {model}")
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
      >>> px.connect(cache_options=px.CacheOptions(cache_path="/tmp/cache"))
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
      >>> px.connect(cache_options=px.CacheOptions(cache_path="/tmp/cache"))
      >>> px.generate_text(prompt="Hello")
      >>> px.reset_state()  # Clear state and cache
      >>> # Next call will create a new client
  """
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    return
  if _DEFAULT_CLIENT.platform_used_for_default_model_cache:
    _DEFAULT_CLIENT.model_cache_manager.clear_cache()
  _DEFAULT_CLIENT = None


class DefaultModelsConnector(ModelConnector):
  """Provides access to model discovery for the default global client.

  This class extends ModelConnector to work with the default global ProxAI
  client. It is typically accessed via the ``px.models`` singleton.

  For client-specific model discovery, use ``client.models`` instead.

  Example:
      >>> import proxai as px
      >>> # Using the default client singleton
      >>> models = px.models.list_models()
      >>> # List working models only
      >>> working = px.models.list_working_models()
  """

  def __init__(self) -> None:
    """Initializes the DefaultModelsConnector with the default client getter."""
    super().__init__(get_default_proxai_client)

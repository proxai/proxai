from __future__ import annotations

from typing import Dict, Any, List

import pydantic

import proxai.chat.chat_session as chat_session
import proxai.chat.message as message_module
import proxai.client as client
import proxai.types as types

# Re-export for easy access
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions
ProviderCallOptions = types.ProviderCallOptions
ModelProbeOptions = types.ModelProbeOptions
DebugOptions = types.DebugOptions
FeatureMappingStrategy = types.FeatureMappingStrategy

Chat = chat_session.Chat
Message = message_module.Message
ProviderModelType = types.ProviderModelType
ParameterType = types.ParameterType
ThinkingType = types.ThinkingType
Tools = types.Tools
OutputFormat = types.OutputFormat
OutputFormatType = types.OutputFormatType
ConnectionOptions = types.ConnectionOptions
MessageRoleType = types.MessageRoleType
ContentType = types.ContentType

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
    provider_call_options: types.ProviderCallOptions | None = None,
    model_probe_options: types.ModelProbeOptions | None = None,
    debug_options: types.DebugOptions | None = None,
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
      provider_call_options: Client-wide defaults for provider call
          behaviour. Controls feature mapping strategy and error
          suppression. See ProviderCallOptions for available settings.
      model_probe_options: Configuration for model probing (health
          checks, model discovery). Controls multiprocessing and
          timeout. See ModelProbeOptions for available settings.
      debug_options: Developer-only diagnostic options. See DebugOptions
          for available settings.

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
      provider_call_options=provider_call_options,
      model_probe_options=model_probe_options,
      debug_options=debug_options,
  )
  if _DEFAULT_CLIENT is not None:
    _DEFAULT_CLIENT.close()
  _DEFAULT_CLIENT = client.ProxAIClient(init_from_params=proxai_client_params)


def generate(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    output_format: types.OutputFormatParam | None = None,
    connection_options: types.ConnectionOptions | None = None,
) -> types.CallRecord:
  return get_default_proxai_client().generate(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      output_format=output_format,
      connection_options=connection_options,
  )


def generate_text(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
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
      >>> import proxai as px
      >>> response = px.generate_text(prompt="What is the capital of France?")
      >>> print(response)
      'The capital of France is Paris.'
  """
  return get_default_proxai_client().generate_text(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      connection_options=connection_options,
  )


def generate_json(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    connection_options: types.ConnectionOptions | None = None,
) -> dict:
  """Generates a JSON response using the configured AI model.

  Thin alias for generate() that resolves the default model, sets
  output_format to JSON, and returns the parsed dict directly.

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
      >>> import proxai as px
      >>> result = px.generate_json(
      ...   prompt="Return the capital of France as JSON"
      ... )
      >>> print(result)
      {'capital': 'Paris', 'country': 'France'}
  """
  return get_default_proxai_client().generate_json(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      connection_options=connection_options,
  )


def generate_pydantic(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    output_format: types.OutputFormatParam | None = None,
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
      output_format: The pydantic model class to validate against.
      connection_options: Connection options.

  Returns:
      An instance of the pydantic model specified in output_format.
      If the provider returns an error and suppress_provider_errors is
      True, returns the error message string.

  Example:
      >>> from pydantic import BaseModel
      >>> class City(BaseModel):
      ...   name: str
      ...   country: str
      >>> import proxai as px
      >>> result = px.generate_pydantic(
      ...   prompt="What is the capital of France?",
      ...   output_format=City
      ... )
      >>> print(result.name)
      'Paris'
  """
  return get_default_proxai_client().generate_pydantic(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      output_format=output_format,
      connection_options=connection_options,
  )


def generate_image(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    connection_options: types.ConnectionOptions | None = None,
):
  """Generates an image using the configured AI model.

  Thin alias for generate() that resolves the default model, sets
  output_format to IMAGE, and returns the image content directly.
  """
  return get_default_proxai_client().generate_image(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      connection_options=connection_options,
  )


def generate_audio(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    connection_options: types.ConnectionOptions | None = None,
):
  """Generates audio using the configured AI model.

  Thin alias for generate() that resolves the default model, sets
  output_format to AUDIO, and returns the audio content directly.
  """
  return get_default_proxai_client().generate_audio(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      connection_options=connection_options,
  )


def generate_video(
    prompt: str | None = None,
    messages: types.MessagesParam | None = None,
    system_prompt: str | None = None,
    provider_model: types.ProviderModelParam | None = None,
    parameters: types.ParameterType | None = None,
    tools: List[types.Tools] | None = None,
    connection_options: types.ConnectionOptions | None = None,
):
  """Generates video using the configured AI model.

  Thin alias for generate() that resolves the default model, sets
  output_format to VIDEO, and returns the video content directly.
  """
  return get_default_proxai_client().generate_video(
      prompt=prompt,
      messages=messages,
      system_prompt=system_prompt,
      provider_model=provider_model,
      parameters=parameters,
      tools=tools,
      connection_options=connection_options,
  )


def set_model(
    provider_model: types.ProviderModelIdentifierType | None = None,
    generate_text: types.ProviderModelIdentifierType | None = None,
    generate_json: types.ProviderModelIdentifierType | None = None,
    generate_pydantic: types.ProviderModelIdentifierType | None = None,
    generate_image: types.ProviderModelIdentifierType | None = None,
    generate_audio: types.ProviderModelIdentifierType | None = None,
    generate_video: types.ProviderModelIdentifierType | None = None,
) -> None:
  """Sets the default model for generation requests.

  Args:
      provider_model: Sets the default TEXT model (backward compat).
      generate_text: Default model for generate_text().
      generate_json: Default model for generate_json().
      generate_pydantic: Default model for generate_pydantic().
      generate_image: Default model for generate_image().
      generate_audio: Default model for generate_audio().
      generate_video: Default model for generate_video().

  Example:
      >>> import proxai as px
      >>> px.set_model(
      ...   generate_text=('openai', 'gpt-4o'),
      ...   generate_image=('openai', 'dall-e-3'),
      ... )
  """
  get_default_proxai_client().set_model(
      provider_model=provider_model,
      generate_text=generate_text,
      generate_json=generate_json,
      generate_pydantic=generate_pydantic,
      generate_image=generate_image,
      generate_audio=generate_audio,
      generate_video=generate_video,
  )


def get_current_options(json: bool = False,) -> types.RunOptions | dict:
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
  _DEFAULT_CLIENT.close()
  _DEFAULT_CLIENT = None


def close() -> None:
  """Release SDK connection pools held by the default client.

  The default client remains configured and usable; subsequent calls
  lazily rebuild SDK pools. Use this when you want to drop file
  descriptors without reconfiguring (for example, between consensus
  rounds in long-running workers). For full teardown including cache
  clearing, use ``reset_state()`` instead.
  """
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    return
  _DEFAULT_CLIENT.close()


class DefaultModelsConnector(client.ModelConnector):
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


class DefaultFilesConnector(client.FileConnector):
  """Provides access to file operations for the default global client.

  This class extends FileConnector to work with the default global ProxAI
  client. It is typically accessed via the ``px.files`` singleton.

  For client-specific file operations, use ``client.files`` instead.

  Example:
      >>> import proxai as px
      >>> px.files.upload(...)
      >>> px.files.list()
  """

  def __init__(self) -> None:
    super().__init__(get_default_proxai_client)

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
import proxai.chat.message_content as message_content
import proxai.caching.model_cache as model_cache
import proxai.caching.query_cache as query_cache
import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.available_models as available_models
import proxai.connectors.files as files_manager
import proxai.connections.proxdash as proxdash
import proxai.connectors.model_configs as model_configs
import proxai.connectors.provider_connector as provider_connector
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
      input_format: types.InputFormatTypeParam | None = None,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      feature_tags: types.FeatureTagParam | None = None,
      tool_tags: types.ToolTagParam | None = None,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType]:
    """Lists all configured models matching the specified criteria.

    Args:
        model_size: Filter by model size category.
        features: Deprecated. Filter by required features.
        input_format: Filter by input format capabilities.
        output_format: Filter by output format capabilities.
        feature_tags: Filter by general feature support.
        tool_tags: Filter by tool support.
        recommended_only: If True, returns only recommended models.
    """
    return self._client_getter().available_models_instance.list_models(
        model_size=model_size,
        features=features,
        input_format=input_format,
        output_format=output_format,
        feature_tags=feature_tags,
        tool_tags=tool_tags,
        recommended_only=recommended_only,
    )

  def list_providers(
      self,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True,
  ) -> list[str]:
    """Lists all providers that have API keys configured.

    Args:
        output_format: Filter by output format. Defaults to TEXT.
        recommended_only: If True, returns only providers with recommended
            models. Defaults to True.
    """
    return self._client_getter().available_models_instance.list_providers(
        output_format=output_format,
        recommended_only=recommended_only,
    )

  def list_provider_models(
      self,
      provider: str,
      model_size: types.ModelSizeIdentifierType | None = None,
      features: types.FeatureTagParam | None = None,
      input_format: types.InputFormatTypeParam | None = None,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      feature_tags: types.FeatureTagParam | None = None,
      tool_tags: types.ToolTagParam | None = None,
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType]:
    """Lists all models available from a specific provider.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_size: Filter by model size category.
        features: Deprecated. Filter by required features.
        input_format: Filter by input format capabilities.
        output_format: Filter by output format capabilities.
        feature_tags: Filter by general feature support.
        tool_tags: Filter by tool support.
        recommended_only: If True, returns only recommended models.
    """
    return (
        self._client_getter().available_models_instance.list_provider_models(
            provider=provider,
            model_size=model_size,
            features=features,
            input_format=input_format,
            output_format=output_format,
            feature_tags=feature_tags,
            tool_tags=tool_tags,
            recommended_only=recommended_only,
        )
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

  def get_model_config(
      self,
      provider: str,
      model: str,
  ) -> types.ProviderModelConfig:
    """Gets the full config for a specific model.

    Returns the ProviderModelConfig including provider model info,
    pricing, features, and metadata.

    Args:
        provider: The provider name (e.g., 'openai', 'gemini').
        model: The model name (e.g., 'gpt-4o', 'gemini-2.5-flash').

    Returns:
        types.ProviderModelConfig: The full model configuration.

    Raises:
        KeyError: If the model doesn't exist.

    Example:
        >>> import proxai as px
        >>> config = px.models.get_model_config("gemini", "gemini-2.5-flash")
        >>> print(config.features.input_format.image)
        SUPPORTED

        >>> # Using a specific client
        >>> client = px.Client()
        >>> config = client.models.get_model_config("gemini", "gemini-2.5-flash")
    """
    return self._client_getter().available_models_instance.get_model_config(
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
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """Lists models that have been verified to be working.

    Args:
        model_size: Filter by model size category.
        features: Filter by required features.
        verbose: If True, prints progress information during testing.
        return_all: If True, returns a ModelStatus object.
        clear_model_cache: If True, clears and retests all models.
        output_format: Output format to test. Defaults to TEXT.
        recommended_only: If True, returns only recommended models.

    Example:
        >>> import proxai as px
        >>> working_models = px.models.list_working_models(verbose=False)
        >>> print(f"Found {len(working_models)} working models")

        >>> # Using a specific client
        >>> client = px.Client()
        >>> working_models = client.models.list_working_models(
        ...   verbose=False)
    """
    return (
        self._client_getter().available_models_instance.list_working_models(
            model_size=model_size,
            features=features,
            verbose=verbose,
            return_all=return_all,
            clear_model_cache=clear_model_cache,
            output_format=output_format,
            recommended_only=recommended_only,
        )
    )

  def list_working_providers(
      self,
      verbose: bool = True,
      clear_model_cache: bool = False,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
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
        output_format: The output format to test. Can be an
            OutputFormatType enum or a string (e.g., 'text').
            Defaults to TEXT.
        recommended_only: If True, returns only providers with
            recommended models curated by ProxAI. Set to False to
            include all available providers. Defaults to True.

    Returns:
        List[str]: A sorted list of provider names with working
            models.

    Example:
        >>> import proxai as px
        >>> providers = px.models.list_working_providers(verbose=False)
        >>> print(providers)
        ['anthropic', 'openai']

        >>> # Using a specific client
        >>> client = px.Client()
        >>> providers = client.models.list_working_providers(
        ...   verbose=False)
    """
    return (
        self._client_getter().available_models_instance.list_working_providers(
            verbose=verbose,
            clear_model_cache=clear_model_cache,
            output_format=output_format,
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
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
      recommended_only: bool = True,
  ) -> list[types.ProviderModelType] | types.ModelStatus:
    """Lists working models from a specific provider.

    Tests models from the specified provider and returns only those
    that successfully respond.

    Args:
        provider: The provider name to list models for.
        model_size: Filter by model size category.
        features: Filter by required features.
        verbose: If True, prints progress information during testing.
            Defaults to True.
        return_all: If True, returns a ModelStatus object with
            detailed results. Defaults to False.
        clear_model_cache: If True, clears the model cache and
            retests models. Defaults to False.
        output_format: The output format to test. Can be an
            OutputFormatType enum or a string (e.g., 'text').
            Defaults to TEXT.
        recommended_only: If True, returns only recommended models
            curated by ProxAI. Set to False to include all available
            models. Defaults to True.

    Returns:
        Union[List[types.ProviderModelType], types.ModelStatus]: A
            list of working ProviderModelType objects from the
            provider, or a ModelStatus object if return_all is True.

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
        >>> openai_working = (
        ...   client.models.list_working_provider_models(
        ...     "openai", verbose=False
        ...   )
        ... )
    """
    available_models = (self._client_getter().available_models_instance)
    return available_models.list_working_provider_models(
        provider=provider,
        model_size=model_size,
        features=features,
        verbose=verbose,
        return_all=return_all,
        clear_model_cache=clear_model_cache,
        output_format=output_format,
        recommended_only=recommended_only,
    )

  def get_working_model(
      self,
      provider: str,
      model: str,
      verbose: bool = False,
      clear_model_cache: bool = False,
      output_format: types.
      OutputFormatTypeParam = (types.OutputFormatType.TEXT),
  ) -> types.ProviderModelType:
    """Verifies and returns a specific model if it's working.

    Tests the specified model and returns it only if it successfully
    responds. Results are cached.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model: The model name (e.g., 'gpt-4', 'claude-3-opus').
        verbose: If True, prints progress information during testing.
            Defaults to False.
        clear_model_cache: If True, clears the model cache and
            retests the model. Defaults to False.
        output_format: The output format to test. Can be an
            OutputFormatType enum or a string (e.g., 'text').
            Defaults to TEXT.

    Returns:
        types.ProviderModelType: The model information if the model
            is working.

    Raises:
        ValueError: If the provider's API key is not found, the
            model doesn't exist, or the model failed the health
            check.

    Example:
        >>> import proxai as px
        >>> model = px.models.get_working_model("openai", "gpt-4")
        >>> print(model)
        (openai, gpt-4)

        >>> # Using a specific client
        >>> client = px.Client()
        >>> model = client.models.get_working_model(
        ...   "openai", "gpt-4")
    """
    return (
        self._client_getter().available_models_instance.get_working_model(
            provider=provider,
            model=model,
            verbose=verbose,
            clear_model_cache=clear_model_cache,
            output_format=output_format,
        )
    )

  def check_health(
      self,
      verbose: bool = True,
  ) -> types.ModelStatus:
    """Tests all models and reports health status.

    Always clears the model cache and retests every model. Uses the
    client's configured model_probe_options for timeout and
    multiprocessing settings.

    Args:
        verbose: If True, prints progress and per-model results.

    Returns:
        ModelStatus with working_models, failed_models, and
        provider_queries.

    Example:
        >>> import proxai as px
        >>> status = px.models.check_health(verbose=False)
        >>> print(f"Working: {len(status.working_models)}")

        >>> client = px.Client(
        ...   model_probe_options=px.ModelProbeOptions(timeout=10))
        >>> status = client.models.check_health()
    """
    return (
        self._client_getter().available_models_instance.check_health(
            verbose=verbose
        )
    )

  def get_default_model_list(self) -> list[types.ProviderModelType]:
    """Returns the default model priority list used for fallback selection.

    Returns:
        list[types.ProviderModelType]: An ordered list of models used
            as the default fallback priority.

    Example:
        >>> import proxai as px
        >>> models = px.models.get_default_model_list()
        >>> print(models[0])
        (openai, gpt-4o)

        >>> # Using a specific client
        >>> client = px.Client()
        >>> models = client.models.get_default_model_list()
    """
    return (
        self._client_getter().model_configs_instance.
        get_default_model_priority_list()
    )


class FileConnector:
  """Provides access to provider File API operations.

  This class offers methods to upload, download, list, and remove
  files via provider File APIs. It can be accessed via the ``px.files``
  singleton for the default client, or via ``client.files`` for a
  specific client instance.

  Example:
      >>> import proxai as px
      >>> px.files.upload(...)
      >>> client = px.Client()
      >>> client.files.list()
  """

  def __init__(
      self,
      client_getter: Callable[[], "ProxAIClient"],
  ) -> None:
    self._client_getter = client_getter

  def upload(
      self,
      media: message_content.MessageContent,
      providers: list[str],
  ) -> message_content.MessageContent:
    """Upload media to specified provider File APIs.

    Uploads the file to each provider in parallel (controlled by
    ``ProviderCallOptions.allow_parallel_file_operations``). On
    success, populates ``media.provider_file_api_status`` and
    ``media.provider_file_api_ids`` with upload metadata and file
    IDs for each provider.

    Supported providers: gemini, claude, openai, mistral.
    Use ``is_upload_supported()`` to check MIME type compatibility.

    Args:
        media: A MessageContent of a media type (IMAGE, DOCUMENT,
            AUDIO, VIDEO). Must have at least ``path`` or ``data``
            set.
        providers: Provider names to upload to (e.g.,
            ``['gemini', 'claude']``).

    Returns:
        The same MessageContent with provider_file_api_status and
        provider_file_api_ids populated.

    Raises:
        ValueError: If media type is not a media content type, if
            ``path``/``data`` are both None, or if a provider is
            not supported or has no API key configured.
        FileUploadError: If one or more provider uploads fail. The
            media object still contains results from successful
            uploads accessible via ``error.media``.

    Example:
        >>> media = px.MessageContent(
        ...     path='report.pdf', media_type='application/pdf')
        >>> px.files.upload(media=media, providers=['gemini', 'claude'])
        >>> print(media.provider_file_api_ids)
        {'gemini': 'files/abc123', 'claude': 'file_xyz789'}
    """
    return self._client_getter().files_manager_instance.upload(
        media=media, providers=providers)

  def download(
      self,
      media: message_content.MessageContent,
      provider: str | None = None,
      path: str | None = None,
  ) -> message_content.MessageContent:
    """Download a file from a provider File API.

    Currently only Mistral supports downloading uploaded files.
    Gemini, Claude, and OpenAI do not support downloading
    user-uploaded files. Use ``is_download_supported()`` to check.

    When ``path`` is provided, the file is saved to disk and
    ``media.path`` is set. When ``path`` is None, the file bytes
    are stored in ``media.data``.

    Args:
        media: A MessageContent with provider_file_api_ids
            populated from a previous ``upload()`` or ``list()``
            call.
        provider: Provider to download from. If None, uses
            priority order (mistral first). Falls back to any
            available provider in media metadata.
        path: Local file path to save to. If None, stores bytes
            in ``media.data`` instead.

    Returns:
        The same MessageContent with ``path`` or ``data``
        populated.

    Raises:
        ValueError: If provider is not found in media metadata,
            or if the provider does not support downloading
            uploaded files.

    Example:
        >>> files = px.files.list(providers=['mistral'])
        >>> px.files.download(media=files[0], path='/tmp/doc.pdf')
        >>> print(files[0].path)
        /tmp/doc.pdf
    """
    return self._client_getter().files_manager_instance.download(
        media=media, provider=provider, path=path)

  def list(
      self,
      providers: list[str] | None = None,
      limit_per_provider: int = 100,
  ) -> list[message_content.MessageContent]:
    """List files from provider File APIs.

    Queries each provider in parallel (controlled by
    ``ProviderCallOptions.allow_parallel_file_operations``) and
    returns a combined flat list. Each returned MessageContent
    represents one file with single-provider metadata.

    The returned MessageContent objects have ``filename``,
    ``media_type``, ``provider_file_api_ids``, and
    ``provider_file_api_status`` populated. ``path`` and ``data``
    are None (use ``download()`` to fetch file contents).

    Args:
        providers: Provider names to query (e.g.,
            ``['gemini', 'openai']``). None queries all providers
            with API keys configured.
        limit_per_provider: Max files to fetch from each provider.
            Defaults to 100. Total results can be up to
            ``limit_per_provider * len(providers)``.

    Returns:
        list[MessageContent]: One per file, each with
        single-provider metadata.

    Raises:
        ValueError: If providers is an empty list or no providers
            have API keys configured.

    Example:
        >>> files = px.files.list(providers=['gemini'])
        >>> for f in files:
        ...     print(f.filename, f.provider_file_api_ids)
        >>> files = px.files.list(limit_per_provider=10)
    """
    return self._client_getter().files_manager_instance.list(
        providers=providers, limit_per_provider=limit_per_provider)

  def remove(
      self,
      media: message_content.MessageContent,
      providers: list[str] | None = None,
  ) -> message_content.MessageContent:
    """Remove uploaded files from provider File APIs.

    Removes files from each provider in parallel (controlled by
    ``ProviderCallOptions.allow_parallel_file_operations``). On
    success, clears the provider's entries from
    ``media.provider_file_api_status`` and
    ``media.provider_file_api_ids``.

    Args:
        media: A MessageContent with provider_file_api_ids
            populated from a previous ``upload()`` call.
        providers: Provider names to remove from (e.g.,
            ``['gemini']``). None removes from all uploaded
            providers. Empty list raises ValueError.

    Returns:
        The same MessageContent with removed providers cleared.

    Raises:
        ValueError: If providers is an empty list, if media has
            no uploaded files, or if a provider has no file_id on
            this media.
        FileRemoveError: If one or more provider removals fail.
            Successfully removed providers are cleared; failed
            providers remain on the media object. Access partial
            results via ``error.media``.

    Example:
        >>> px.files.remove(media=media, providers=['gemini'])
        >>> px.files.remove(media=media)  # removes from all
        >>> print(media.provider_file_api_ids)  # {}
    """
    return self._client_getter().files_manager_instance.remove(
        media=media, providers=providers)

  def is_upload_supported(
      self,
      media: message_content.MessageContent,
      provider: str,
  ) -> bool:
    """Check if a media file can be uploaded to a provider's File API.

    Gemini, Claude, and OpenAI support all media types. Mistral
    supports documents (PDF, DOCX, XLSX, CSV, TXT) and images only.

    Args:
        media: A MessageContent with media_type set.
        provider: Provider name (e.g., 'gemini', 'mistral').

    Returns:
        True if the provider supports uploading this media type.

    Example:
        >>> media = px.MessageContent(
        ...     path='video.mp4', media_type='video/mp4')
        >>> px.files.is_upload_supported(media, 'gemini')   # True
        >>> px.files.is_upload_supported(media, 'mistral')  # False
    """
    return self._client_getter().files_manager_instance.is_upload_supported(
        media=media, provider=provider)

  def is_download_supported(self, provider: str) -> bool:
    """Check if a provider supports downloading uploaded files.

    Currently only Mistral supports downloading user-uploaded
    files. Gemini, Claude, and OpenAI do not allow downloading
    files that were uploaded via their File APIs.

    Args:
        provider: Provider name (e.g., 'gemini', 'mistral').

    Returns:
        True if the provider supports downloading uploaded files.

    Example:
        >>> px.files.is_download_supported('mistral')  # True
        >>> px.files.is_download_supported('gemini')   # False
    """
    return self._client_getter().files_manager_instance.is_download_supported(
        provider=provider)


@dataclasses.dataclass
class ProxAIClientParams:
  """Initialization parameters for ProxAIClient."""

  experiment_path: str | None = None
  cache_options: types.CacheOptions | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_options: types.ProxDashOptions | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  model_probe_options: types.ModelProbeOptions | None = None
  debug_options: types.DebugOptions | None = None


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
      provider_call_options: types.ProviderCallOptions | None = None,
      model_probe_options: types.ModelProbeOptions | None = None,
      debug_options: types.DebugOptions | None = None,
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
        provider_call_options: Client-wide defaults for provider call
            behaviour. Controls feature mapping strategy and error
            suppression. See ProviderCallOptions for available settings.
        model_probe_options: Configuration for model probing (health
            checks, model discovery). Controls multiprocessing and
            timeout. See ModelProbeOptions for available settings.
        debug_options: Developer-only diagnostic options. See DebugOptions
            for available settings.
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
        ...   provider_call_options=px.ProviderCallOptions(
        ...     suppress_provider_errors=True,
        ...   ),
        ... )
        >>> response = client.generate_text(
        ...   prompt="What is the capital of France?"
        ... )
    """
    if init_from_params is not None or init_from_state is not None:
      if (
          experiment_path is not None or cache_options is not None or
          logging_options is not None or proxdash_options is not None or
          provider_call_options is not None or
          model_probe_options is not None or debug_options is not None
      ):
        raise ValueError(
            "init_from_params or init_from_state cannot be set at with "
            "direct arguments. Please use one of init_from_params, "
            "init_from_state, or direct arguments.\n"
            f"experiment_path: {experiment_path}\n"
            f"cache_options: {cache_options}\n"
            f"logging_options: {logging_options}\n"
            f"proxdash_options: {proxdash_options}\n"
            f"provider_call_options: {provider_call_options}\n"
            f"model_probe_options: {model_probe_options}\n"
            f"debug_options: {debug_options}\n"
            f"init_from_params: {init_from_params}\n"
            f"init_from_state: {init_from_state}\n"
        )
    else:
      init_from_params = ProxAIClientParams(
          experiment_path=experiment_path,
          cache_options=cache_options,
          logging_options=logging_options,
          proxdash_options=proxdash_options,
          provider_call_options=provider_call_options,
          model_probe_options=model_probe_options,
          debug_options=debug_options,
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
      self.provider_call_options = init_from_params.provider_call_options
      self.model_probe_options = init_from_params.model_probe_options
      self.debug_options = init_from_params.debug_options

      _ = self.model_cache_manager
      _ = self.query_cache_manager
      self._init_proxdash_connection()

      if self.cache_options and self.cache_options.clear_model_cache_on_connect:
        self.model_cache_manager.clear_cache()
      if self.cache_options and self.cache_options.clear_query_cache_on_connect:
        self.query_cache_manager.clear_cache()

      api_key_manager_params = api_key_manager.ApiKeyManagerParams(
          proxdash_connection=self.proxdash_connection,
      )
      self._api_key_manager_instance = api_key_manager.ApiKeyManager(
          init_from_params=api_key_manager_params
      )

      files_manager_params = files_manager.FilesManagerParams(
          run_type=self.run_type,
          logging_options=self.logging_options,
          proxdash_connection=self.proxdash_connection,
          provider_call_options=self.provider_call_options,
          api_key_manager=self._api_key_manager_instance,
      )
      self._files_manager_instance = files_manager.FilesManager(
          init_from_params=files_manager_params
      )

      available_models_params = available_models.AvailableModelsParams(
          run_type=self.run_type,
          provider_call_options=self.provider_call_options,
          model_configs_instance=self.model_configs_instance,
          model_cache_manager=self.model_cache_manager,
          query_cache_manager=self.query_cache_manager,
          logging_options=self.logging_options,
          proxdash_connection=self.proxdash_connection,
          api_key_manager=self._api_key_manager_instance,
          files_manager=self._files_manager_instance,
          model_probe_options=self.model_probe_options,
          debug_options=self.debug_options,
      )
      self._available_models_instance = available_models.AvailableModels(
          init_from_params=available_models_params
      )

    self._validate_raw_provider_response_options()
    self._maybe_emit_raw_provider_response_warning()

  def get_internal_state_property_name(self) -> str:
    """Return the name of the internal state property."""
    return _PROXAI_CLIENT_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
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

    self.provider_call_options = None
    self.model_probe_options = None
    self.debug_options = None

    self.api_key_manager_instance = None
    self.available_models_instance = None
    self.files_manager_instance = None

  def _validate_raw_provider_response_options(self):
    """Reject keep_raw_provider_response=True while a query cache is set.

    Runs after both the direct-kwargs and load_state branches of __init__
    converge, so it protects every construction path — including state
    restoration, which goes through
    set_property_value_without_triggering_getters and would otherwise
    bypass any validation living in the property setter.
    """
    if (
        self.debug_options.keep_raw_provider_response and
        self.cache_options is not None
    ):
      raise ValueError(
          "keep_raw_provider_response=True is incompatible with "
          "cache_options. The query cache cannot reconstruct provider "
          "SDK objects on cache hits, so the combination would silently "
          "return None for cached calls. To use both, construct two "
          "clients: a cached production client and a separate debug "
          "client with keep_raw_provider_response=True."
      )

  def _maybe_emit_raw_provider_response_warning(self):
    if not self.debug_options.keep_raw_provider_response:
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
  def platform_used_for_default_model_cache(self) -> bool | None:
    return self.get_property_value("platform_used_for_default_model_cache")

  @platform_used_for_default_model_cache.setter
  def platform_used_for_default_model_cache(self, value: bool | None):
    self.set_property_value("platform_used_for_default_model_cache", value)

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
  def registered_model_connectors(self) -> dict | None:
    return self.get_property_value("registered_model_connectors")

  @registered_model_connectors.setter
  def registered_model_connectors(self, value: dict | None):
    self.set_property_value("registered_model_connectors", value)

  @property
  def provider_call_options(self) -> types.ProviderCallOptions:
    return self.get_property_value("provider_call_options")

  @provider_call_options.setter
  def provider_call_options(
      self,
      value: types.ProviderCallOptions | None,
  ):
    result = types.ProviderCallOptions()
    if value is not None:
      result.feature_mapping_strategy = value.feature_mapping_strategy
      result.suppress_provider_errors = value.suppress_provider_errors
      result.allow_parallel_file_operations = (
          value.allow_parallel_file_operations)
    self.set_property_value("provider_call_options", result)

  @property
  def model_probe_options(self) -> types.ModelProbeOptions:
    return self.get_property_value("model_probe_options")

  @model_probe_options.setter
  def model_probe_options(
      self,
      value: types.ModelProbeOptions | None,
  ):
    result = types.ModelProbeOptions()
    if value is not None:
      result.allow_multiprocessing = value.allow_multiprocessing
      if value.timeout < 1:
        raise ValueError("ModelProbeOptions.timeout must be >= 1.")
      result.timeout = value.timeout
    self.set_property_value("model_probe_options", result)

  @property
  def debug_options(self) -> types.DebugOptions:
    return self.get_property_value("debug_options")

  @debug_options.setter
  def debug_options(self, value: types.DebugOptions | None):
    result = types.DebugOptions()
    if value is not None:
      result.keep_raw_provider_response = value.keep_raw_provider_response
    self.set_property_value("debug_options", result)

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

  def model_configs_instance_deserializer(
      self, state_value: types.ModelConfigsState
  ) -> model_configs.ModelConfigs:
    return model_configs.ModelConfigs(init_from_state=state_value)

  def default_model_cache_manager_deserializer(
      self, state_value: types.ModelCacheManagerState
  ) -> model_cache.ModelCacheManager:
    return model_cache.ModelCacheManager(init_from_state=state_value)

  def model_cache_manager_deserializer(
      self, state_value: types.ModelCacheManagerState
  ) -> model_cache.ModelCacheManager:
    return model_cache.ModelCacheManager(init_from_state=state_value)

  def query_cache_manager_deserializer(
      self, state_value: types.QueryCacheManagerState
  ) -> query_cache.QueryCacheManager:
    return query_cache.QueryCacheManager(init_from_state=state_value)

  def proxdash_connection_deserializer(
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def api_key_manager_instance(self) -> api_key_manager.ApiKeyManager:
    return self.get_state_controlled_property_value(
        "api_key_manager_instance")

  @api_key_manager_instance.setter
  def api_key_manager_instance(self, value: api_key_manager.ApiKeyManager):
    self.set_state_controlled_property_value(
        "api_key_manager_instance", value)

  def api_key_manager_instance_deserializer(
      self, state_value: types.ApiKeyManagerState
  ) -> api_key_manager.ApiKeyManager:
    return api_key_manager.ApiKeyManager(init_from_state=state_value)

  def available_models_instance_deserializer(
      self, state_value: types.AvailableModelsState
  ) -> available_models.AvailableModels:
    return available_models.AvailableModels(init_from_state=state_value)

  @property
  def files_manager_instance(self) -> files_manager.FilesManager:
    return self.get_state_controlled_property_value(
        "files_manager_instance")

  @files_manager_instance.setter
  def files_manager_instance(self, value: files_manager.FilesManager):
    self.set_state_controlled_property_value(
        "files_manager_instance", value)

  def files_manager_instance_deserializer(
      self, state_value: types.FilesManagerState
  ) -> files_manager.FilesManager:
    return files_manager.FilesManager(init_from_state=state_value)

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

  @property
  def files(self) -> FileConnector:
    """Access provider File API operations.

    Provides an interface for uploading, downloading, listing, and
    removing files via provider File APIs. This property returns a
    FileConnector instance bound to this client.

    Example:
        >>> client = px.Client()
        >>> client.files.upload(...)
        >>> client.files.list()
    """
    if not hasattr(self, "_files_connector") or self._files_connector is None:
      self._files_connector = FileConnector(lambda: self)
    return self._files_connector

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

  _OUTPUT_FORMAT_TYPE_FALLBACK = {
      types.OutputFormatType.PYDANTIC: [
          types.OutputFormatType.JSON, types.OutputFormatType.TEXT
      ],
      types.OutputFormatType.JSON: [types.OutputFormatType.TEXT],
  }

  def get_default_provider_model(
      self,
      output_format_type: types.OutputFormatType = None,
  ) -> types.ProviderModelType:
    """Resolve and return the default ProviderModelType."""
    if output_format_type is None:
      output_format_type = types.OutputFormatType.TEXT

    if output_format_type in self.registered_models:
      return self.registered_models[output_format_type]

    fallbacks = self._OUTPUT_FORMAT_TYPE_FALLBACK.get(output_format_type, [])
    for fallback_type in fallbacks:
      if fallback_type in self.registered_models:
        return self.registered_models[fallback_type]

    if output_format_type not in (
        types.OutputFormatType.TEXT, types.OutputFormatType.JSON,
        types.OutputFormatType.PYDANTIC
    ):
      raise ValueError(
          f"Output format type not supported: "
          f"{output_format_type}"
      )

    default_models = (
        self.model_configs_instance.get_default_model_priority_list()
    )
    for provider_model in default_models:
      try:
        self.available_models_instance.get_working_model(
            provider=provider_model.provider, model=provider_model.model
        )
        self.registered_models[types.OutputFormatType.TEXT] = provider_model
        self.registered_model_connectors[types.OutputFormatType.TEXT] = (
            self.available_models_instance.get_model_connector(provider_model)
        )
        return provider_model
      except ValueError:
        continue

    models = self.available_models_instance.list_working_models(return_all=True)
    if len(models.working_models) > 0:
      provider_model = models.working_models.pop()
      self.registered_models[types.OutputFormatType.TEXT] = provider_model
      self.registered_model_connectors[
          types.OutputFormatType.TEXT
      ] = (self.available_models_instance.get_model_connector(provider_model))
      return provider_model

    raise ValueError(
        "No working models found in current environment:\n"
        "* Please check your environment variables and try "
        "again.\n"
        "* You can use px.models.check_health() method as instructed "
        "in https://www.proxai.co/proxai-docs/check-health"
    )

  def get_registered_model_connector(
      self,
      output_format_type: types.OutputFormatType = None,
  ) -> provider_connector.ProviderConnector:
    """Get or create a connector for the default model."""
    if output_format_type is None:
      output_format_type = types.OutputFormatType.TEXT
    self.get_default_provider_model(output_format_type=output_format_type)

    if output_format_type in self.registered_model_connectors:
      return self.registered_model_connectors[output_format_type]
    fallbacks = self._OUTPUT_FORMAT_TYPE_FALLBACK.get(output_format_type, [])
    for fallback_type in fallbacks:
      if fallback_type in self.registered_model_connectors:
        return self.registered_model_connectors[fallback_type]
    raise ValueError(
        "No registered model connector for output format "
        f"type: {output_format_type}"
    )

  def _register_model_for_output_format_type(
      self,
      output_format_type: types.OutputFormatType,
      provider_model: types.ProviderModelIdentifierType,
  ):
    self.model_configs_instance.check_provider_model_identifier_type(
        provider_model
    )
    resolved = self.model_configs_instance.get_provider_model(provider_model)
    self.registered_models[output_format_type] = resolved
    self.registered_model_connectors[output_format_type] = (
        self.available_models_instance.get_model_connector(provider_model)
    )

  def set_model(
      self,
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

    Raises:
        ValueError: If provider_model and generate_text are both provided,
            or if no arguments are provided.
    """
    if provider_model and generate_text:
      raise ValueError(
          "provider_model and generate_text cannot be set at the "
          "same time. Please set one of them."
      )

    if generate_text is None and provider_model is not None:
      generate_text = provider_model

    has_any = any([
        generate_text, generate_json, generate_pydantic, generate_image,
        generate_audio, generate_video
    ])
    if not has_any:
      raise ValueError(
          "At least one model must be specified. Use provider_model, "
          "generate_text, generate_json, generate_pydantic, "
          "generate_image, generate_audio, or generate_video."
      )

    mapping = {
        types.OutputFormatType.TEXT: generate_text,
        types.OutputFormatType.JSON: generate_json,
        types.OutputFormatType.PYDANTIC: generate_pydantic,
        types.OutputFormatType.IMAGE: generate_image,
        types.OutputFormatType.AUDIO: generate_audio,
        types.OutputFormatType.VIDEO: generate_video,
    }
    for fmt_type, model in mapping.items():
      if model is not None:
        self._register_model_for_output_format_type(fmt_type, model)

  def generate(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      output_format: types.OutputFormatParam | None = None,
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
          '        ...])'
      )

    if (
        connection_options and connection_options.fallback_models and
        connection_options.suppress_provider_errors
    ):
      raise ValueError(
          'suppress_provider_errors and fallback_models cannot be '
          'used together.\n'
          f'connection_options: {connection_options}'
      )

    if (
        connection_options and connection_options.endpoint and
        connection_options.fallback_models
    ):
      raise ValueError(
          'endpoint and fallback_models cannot be used together.\n'
          f'connection_options: {connection_options}'
      )

    if connection_options is None:
      connection_options = types.ConnectionOptions()
    if connection_options.suppress_provider_errors is None:
      connection_options.suppress_provider_errors = (
          self.provider_call_options.suppress_provider_errors
      )

    if (
        connection_options.override_cache_value and (
            self.query_cache_manager is None or self.query_cache_manager.status
            != types.QueryCacheManagerStatus.WORKING
        )
    ):
      raise ValueError(
          "override_cache_value is True but query cache is not configured.\n"
          "Please set cache_options to enable query cache."
      )

    messages = type_utils.messages_param_to_chat(messages)
    output_format = type_utils.output_format_param_to_output_format(
        output_format
    )

    provider_models = [
        self.model_configs_instance.get_provider_model(provider_model)
    ]

    if connection_options.fallback_models:
      for fallback_model in connection_options.fallback_models:
        provider_models.append(
            self.model_configs_instance.get_provider_model(fallback_model)
        )
      connection_options.suppress_provider_errors = True
      connection_options.fallback_models = None

    connection_metadata = types.ConnectionMetadata(
        feature_mapping_strategy=(
            self.provider_call_options.feature_mapping_strategy
        )
    )
    for idx, provider_model in enumerate(provider_models):
      model_connector = self.available_models_instance.get_model_connector(
          provider_model_identifier=provider_model
      )
      provider_model_config = (
          self.model_configs_instance.get_provider_model_config(provider_model)
      )
      result_record = model_connector.generate(
          prompt=prompt,
          messages=messages,
          system_prompt=system_prompt,
          provider_model=provider_model,
          provider_model_config=provider_model_config,
          parameters=parameters,
          tools=tools,
          output_format=output_format,
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
          output_format_type=types.OutputFormatType.TEXT
      )

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
        >>> client = px.Client()
        >>> result = client.generate_json(
        ...   prompt="Return the capital of France as JSON"
        ... )
        >>> print(result)
        {'capital': 'Paris', 'country': 'France'}
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          output_format_type=types.OutputFormatType.JSON
      )

    output_format = types.OutputFormat(type=types.OutputFormatType.JSON)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        output_format=output_format,
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
        >>> client = px.Client()
        >>> result = client.generate_pydantic(
        ...   prompt="What is the capital of France?",
        ...   output_format=City
        ... )
        >>> print(result.name)
        'Paris'
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          output_format_type=types.OutputFormatType.PYDANTIC
      )

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        output_format=output_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_pydantic

  def generate_image(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> types.MessageContent | str:
    """Generates an image using the configured AI model.

    Thin alias for generate() that resolves the default model, sets
    output_format to IMAGE, and returns the image content directly.

    Args:
        prompt: Text prompt describing the desired image.
        messages: Structured messages for multi-turn conversations.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request.
        parameters: Generation parameters.
        tools: Tools to enable for this request.
        connection_options: Connection options.

    Returns:
        The generated image content. If the provider returns an error
        and suppress_provider_errors is True, returns the error message
        string.
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          output_format_type=types.OutputFormatType.IMAGE
      )

    output_format = types.OutputFormat(type=types.OutputFormatType.IMAGE)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        output_format=output_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_image

  def generate_audio(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> types.MessageContent | str:
    """Generates audio using the configured AI model.

    Thin alias for generate() that resolves the default model, sets
    output_format to AUDIO, and returns the audio content directly.

    Args:
        prompt: Text prompt describing the desired audio.
        messages: Structured messages for multi-turn conversations.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request.
        parameters: Generation parameters.
        tools: Tools to enable for this request.
        connection_options: Connection options.

    Returns:
        The generated audio content. If the provider returns an error
        and suppress_provider_errors is True, returns the error message
        string.
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          output_format_type=types.OutputFormatType.AUDIO
      )

    output_format = types.OutputFormat(type=types.OutputFormatType.AUDIO)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        output_format=output_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_audio

  def generate_video(
      self,
      prompt: str | None = None,
      messages: types.MessagesParam | None = None,
      system_prompt: str | None = None,
      provider_model: types.ProviderModelParam | None = None,
      parameters: types.ParameterType | None = None,
      tools: List[types.ToolType] | None = None,
      connection_options: types.ConnectionOptions | None = None,
  ) -> types.MessageContent | str:
    """Generates video using the configured AI model.

    Thin alias for generate() that resolves the default model, sets
    output_format to VIDEO, and returns the video content directly.

    Args:
        prompt: Text prompt describing the desired video.
        messages: Structured messages for multi-turn conversations.
        system_prompt: System message to set the AI's behavior and context.
        provider_model: Specific provider and model to use for this request.
        parameters: Generation parameters.
        tools: Tools to enable for this request.
        connection_options: Connection options.

    Returns:
        The generated video content. If the provider returns an error
        and suppress_provider_errors is True, returns the error message
        string.
    """
    if provider_model is None:
      provider_model = self.get_default_provider_model(
          output_format_type=types.OutputFormatType.VIDEO
      )

    output_format = types.OutputFormat(type=types.OutputFormatType.VIDEO)

    call_record = self.generate(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
        provider_model=provider_model,
        parameters=parameters,
        tools=tools,
        output_format=output_format,
        connection_options=connection_options,
    )

    if call_record.result.status == types.ResultStatusType.FAILED:
      return call_record.result.error

    return call_record.result.output_video

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
        provider_call_options=self.provider_call_options,
        model_probe_options=self.model_probe_options,
        debug_options=self.debug_options,
    )
    if json:
      return type_serializer.encode_run_options(run_options=run_options)
    return run_options

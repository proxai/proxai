"""File management for provider File APIs."""

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import proxai.chat.message_content as message_content
import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.proxdash as proxdash
import proxai.connectors.file_helpers as file_helpers
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_FILES_MANAGER_STATE_PROPERTY = '_files_manager_state'


class FileUploadError(Exception):
  """Raised when one or more provider uploads fail."""

  def __init__(
      self, errors: dict[str, Exception], media: message_content.MessageContent
  ):
    self.errors = errors
    self.media = media
    providers = ', '.join(errors.keys())
    super().__init__(f"Upload failed for providers: {providers}")


class FileRemoveError(Exception):
  """Raised when one or more provider file removals fail."""

  def __init__(
      self, errors: dict[str, Exception], media: message_content.MessageContent
  ):
    self.errors = errors
    self.media = media
    providers = ', '.join(errors.keys())
    super().__init__(f"Remove failed for providers: {providers}")


@dataclasses.dataclass
class FilesManagerParams:
  """Initialization parameters for FilesManager."""

  run_type: types.RunType | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_call_options: types.ProviderCallOptions | None = None
  api_key_manager: api_key_manager.ApiKeyManager | None = None


class FilesManager(state_controller.StateControlled):
  """Manages file uploads and references across provider File APIs."""

  _DOWNLOAD_PROVIDER_PRIORITY = ['mistral']

  _files_manager_state: types.FilesManagerState

  def __init__(
      self, init_from_params: FilesManagerParams | None = None,
      init_from_state: types.FilesManagerState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.run_type = init_from_params.run_type
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.provider_call_options = init_from_params.provider_call_options
      self.api_key_manager = init_from_params.api_key_manager

  def get_internal_state_property_name(self) -> str:
    return _FILES_MANAGER_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
    return types.FilesManagerState

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
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(
      self, proxdash_connection: proxdash.ProxDashConnection
  ):
    self.set_state_controlled_property_value(
        'proxdash_connection', proxdash_connection
    )

  def proxdash_connection_deserializer(
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def provider_call_options(self) -> types.ProviderCallOptions:
    return self.get_property_value('provider_call_options')

  @provider_call_options.setter
  def provider_call_options(
      self, provider_call_options: types.ProviderCallOptions
  ):
    self.set_property_value('provider_call_options', provider_call_options)

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

  _MEDIA_TYPES = (
      message_content.ContentType.IMAGE,
      message_content.ContentType.DOCUMENT,
      message_content.ContentType.AUDIO,
      message_content.ContentType.VIDEO,
  )

  def _use_parallel(self, providers: list[types.ProviderNameType]) -> bool:
    return (
        len(providers) > 1 and self.provider_call_options is not None and
        self.provider_call_options.allow_parallel_file_operations
    )

  def _validate_provider_support(
      self,
      provider: types.ProviderNameType,
      dispatch: dict,
  ):
    if provider not in dispatch:
      raise ValueError(
          f"Provider '{provider}' does not support the File API. "
          f"Supported: {list(dispatch.keys())}"
      )
    if not self.api_key_manager.has_provider_key(provider):
      raise ValueError(f"No API key configured for provider '{provider}'.")

  def _get_upload_dispatch(self):
    if self.run_type == types.RunType.TEST:
      return file_helpers.MOCK_UPLOAD_DISPATCH
    return file_helpers.UPLOAD_DISPATCH

  def _get_remove_dispatch(self):
    if self.run_type == types.RunType.TEST:
      return file_helpers.MOCK_REMOVE_DISPATCH
    return file_helpers.REMOVE_DISPATCH

  def _get_list_dispatch(self):
    if self.run_type == types.RunType.TEST:
      return file_helpers.MOCK_LIST_DISPATCH
    return file_helpers.LIST_DISPATCH

  def _get_download_dispatch(self):
    if self.run_type == types.RunType.TEST:
      return file_helpers.MOCK_DOWNLOAD_DISPATCH
    return file_helpers.DOWNLOAD_DISPATCH

  # --- Capability checks ---

  def is_upload_supported(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType,
  ) -> bool:
    """Check if a media file can be uploaded and referenced by file_id.

    Checks both whether the File API accepts the upload AND whether
    the generate endpoint accepts the file_id reference for this
    media type. Returns False if either check fails — uploading a
    file that can't be referenced is wasteful.

    Args:
      media: A MessageContent with media_type set.
      provider: Provider name to check.

    Returns:
      True if the provider supports uploading and referencing this
      media type by file_id in generate.
    """
    if media.media_type is None:
      return False
    if provider not in file_helpers.UPLOAD_SUPPORTED_MEDIA_TYPES:
      return False
    if media.media_type not in (
        file_helpers.UPLOAD_SUPPORTED_MEDIA_TYPES[provider]
    ):
      return False
    if provider not in file_helpers.REFERENCE_SUPPORTED_MEDIA_TYPES:
      return False
    return media.media_type in (
        file_helpers.REFERENCE_SUPPORTED_MEDIA_TYPES[provider]
    )

  def is_download_supported(
      self,
      provider: types.ProviderNameType,
  ) -> bool:
    """Check if a provider supports downloading uploaded files.

    Args:
      provider: Provider name to check.

    Returns:
      True if the provider supports downloading uploaded files.
    """
    return provider in file_helpers.DOWNLOAD_SUPPORTED_PROVIDERS

  # --- Upload ---

  def _validate_upload_media(self, media: message_content.MessageContent):
    if media.type not in self._MEDIA_TYPES:
      raise ValueError(
          f"upload() requires a media content type "
          f"(IMAGE, DOCUMENT, AUDIO, VIDEO), got '{media.type.value}'."
      )
    if media.path is None and media.data is None:
      raise ValueError(
          "MessageContent must have 'path' or 'data' set for upload."
      )

  def _resolve_upload_file_info(
      self, media: message_content.MessageContent
  ) -> tuple[str | None, bytes | None, str, str]:
    file_path = media.path
    file_data = media.data
    filename = 'file'
    if file_path:
      filename = os.path.basename(file_path)
    mime_type = media.media_type or 'application/octet-stream'
    return file_path, file_data, filename, mime_type

  def _init_upload_fields(self, media: message_content.MessageContent):
    if media.provider_file_api_status is None:
      media.provider_file_api_status = {}
    if media.provider_file_api_ids is None:
      media.provider_file_api_ids = {}

  def _upload_single_provider(
      self,
      provider: types.ProviderNameType,
      file_path: str | None,
      file_data: bytes | None,
      filename: str,
      mime_type: str,
  ) -> message_content.FileUploadMetadata:
    token_map = self.api_key_manager.get_provider_keys(provider)
    upload_fn = self._get_upload_dispatch()[provider]
    return upload_fn(
        file_path=file_path,
        file_data=file_data,
        filename=filename,
        mime_type=mime_type,
        token_map=token_map,
    )

  def _collect_upload_result(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType,
      metadata: message_content.FileUploadMetadata,
  ):
    metadata.provider = provider
    media.provider_file_api_status[provider] = metadata
    media.provider_file_api_ids[provider] = metadata.file_id

  def _collect_upload_error(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType,
      errors: dict[str, Exception],
      error: Exception,
  ):
    media.provider_file_api_status[provider] = (
        message_content.FileUploadMetadata(
            file_id='',
            provider=provider,
            state=message_content.FileUploadState.FAILED,
        )
    )
    errors[provider] = error

  def _proxdash_connected(self) -> bool:
    return (
        self.proxdash_connection is not None and self.proxdash_connection.status
        == types.ProxDashConnectionStatus.CONNECTED
    )

  def _execute_uploads(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
      file_path: str | None,
      file_data: bytes | None,
      filename: str,
      mime_type: str,
  ) -> dict[str, Exception]:
    """Upload to all providers and ProxDash in parallel."""
    errors: dict[str, Exception] = {}
    include_proxdash = self._proxdash_connected()
    use_parallel = self._use_parallel(providers) or include_proxdash

    if use_parallel:
      max_workers = len(providers) + (1 if include_proxdash else 0)
      with ThreadPoolExecutor(max_workers=max_workers) as pool:
        provider_futures = {
            provider:
                pool.submit(
                    self._upload_single_provider,
                    provider,
                    file_path,
                    file_data,
                    filename,
                    mime_type,
                ) for provider in providers
        }
        proxdash_future = None
        if include_proxdash:
          proxdash_future = pool.submit(
              self.proxdash_connection.upload_file, media
          )

        for provider, future in provider_futures.items():
          try:
            metadata = future.result()
            self._collect_upload_result(media, provider, metadata)
          except Exception as e:
            self._collect_upload_error(media, provider, errors, e)

        if proxdash_future is not None:
          try:
            proxdash_future.result()
          except Exception:
            pass
    else:
      for provider in providers:
        try:
          metadata = self._upload_single_provider(
              provider, file_path, file_data, filename, mime_type
          )
          self._collect_upload_result(media, provider, metadata)
        except Exception as e:
          self._collect_upload_error(media, provider, errors, e)

    self._sync_provider_metadata_to_proxdash(media)
    return errors

  def _sync_provider_metadata_to_proxdash(
      self, media: message_content.MessageContent
  ):
    if not self._proxdash_connected() or media.proxdash_file_id is None:
      return
    try:
      self.proxdash_connection.update_file(media)
    except Exception:
      pass

  def upload(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
  ) -> message_content.MessageContent:
    """Upload media to specified provider File APIs.

    Args:
      media: A MessageContent of a media type (IMAGE, DOCUMENT, AUDIO,
        VIDEO). Must have at least one of path or data set.
      providers: List of provider names to upload to (e.g.,
        ['gemini', 'claude']).

    Returns:
      The same MessageContent with provider_file_api_status and
      provider_file_api_ids populated.

    Raises:
      ValueError: If media type is not a media content type, or if a
        provider is not supported or has no API key configured.
      FileUploadError: If one or more provider uploads fail. The
        media object still contains results from successful uploads.
    """
    self._validate_upload_media(media)
    if not providers and not self._proxdash_connected():
      raise ValueError(
          "No providers specified and ProxDash is not connected. "
          "Provide at least one provider or connect to ProxDash."
      )
    for provider in providers:
      self._validate_provider_support(provider, self._get_upload_dispatch())
      if not self.is_upload_supported(media, provider):
        raise ValueError(
            f"Media type '{media.media_type}' cannot be uploaded "
            f"and referenced by file_id on provider '{provider}'."
        )

    file_path, file_data, filename, mime_type = (
        self._resolve_upload_file_info(media)
    )
    self._init_upload_fields(media)

    errors = self._execute_uploads(
        media, providers, file_path, file_data, filename, mime_type
    )
    if errors:
      raise FileUploadError(errors=errors, media=media)

    return media

  # --- Download ---

  def _resolve_download_provider(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType | None,
  ) -> types.ProviderNameType:
    if provider is not None:
      if (
          media.provider_file_api_ids is None or
          provider not in media.provider_file_api_ids
      ):
        raise ValueError(
            f"No file_id found for provider '{provider}' "
            f"on this media content."
        )
      return provider
    if (media.provider_file_api_ids is None or not media.provider_file_api_ids):
      raise ValueError("No uploaded providers found on this media content.")
    for p in self._DOWNLOAD_PROVIDER_PRIORITY:
      if p in media.provider_file_api_ids:
        return p
    return next(iter(media.provider_file_api_ids))

  def _download_from_proxdash(
      self, media: message_content.MessageContent
  ) -> bytes | None:
    if not self._proxdash_connected() or media.proxdash_file_id is None:
      return None
    try:
      return self.proxdash_connection.download_file(media.proxdash_file_id)
    except Exception:
      return None

  def _write_download_result(
      self,
      media: message_content.MessageContent,
      data: bytes,
      path: str | None,
  ):
    if path is not None:
      with open(path, 'wb') as f:
        f.write(data)
      media.path = path
    else:
      media.data = data

  def download(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType | None = None,
      path: str | None = None,
  ) -> message_content.MessageContent:
    """Download a file, trying ProxDash first then provider File APIs.

    When ProxDash is connected and the media has a proxdash_file_id,
    downloads from ProxDash's S3 storage first. Falls back to provider
    download if ProxDash is unavailable or fails.

    Args:
      media: A MessageContent with proxdash_file_id or
        provider_file_api_ids populated.
      provider: Provider to download from. If None, tries ProxDash
        first, then provider priority order (mistral first, then any
        available provider).
      path: Local file path to save to. If None, stores bytes in
        media.data instead.

    Returns:
      The same MessageContent with path or data populated.

    Raises:
      ValueError: If no download source is available.
    """
    if provider is None:
      data = self._download_from_proxdash(media)
      if data is not None:
        self._write_download_result(media, data, path)
        return media

    provider = self._resolve_download_provider(media, provider)
    download_dispatch = self._get_download_dispatch()
    if provider not in download_dispatch:
      raise ValueError(
          f"Provider '{provider}' does not support file download. "
          f"Supported: "
          f"{list(download_dispatch.keys())}"
      )
    if not self.api_key_manager.has_provider_key(provider):
      raise ValueError(f"No API key configured for provider '{provider}'.")

    file_id = media.provider_file_api_ids[provider]
    token_map = self.api_key_manager.get_provider_keys(provider)
    download_fn = download_dispatch[provider]
    data = download_fn(file_id=file_id, token_map=token_map)

    self._write_download_result(media, data, path)
    return media

  # --- List ---

  def _resolve_list_providers(
      self,
      providers: list[types.ProviderNameType] | None,
  ) -> list[types.ProviderNameType]:
    if providers is not None and len(providers) == 0:
      raise ValueError(
          "'providers' must contain at least one provider name, "
          "or be omitted to list from all available providers."
      )
    if providers is None:
      providers = [
          p for p in self._get_list_dispatch()
          if self.api_key_manager.has_provider_key(p)
      ]
      if not providers:
        raise ValueError("No providers with API keys found for file listing.")
    return providers

  def _list_single_provider(
      self,
      provider: types.ProviderNameType,
      limit: int,
  ) -> list[message_content.FileUploadMetadata]:
    token_map = self.api_key_manager.get_provider_keys(provider)
    list_fn = self._get_list_dispatch()[provider]
    return list_fn(token_map=token_map, limit=limit)

  def _metadata_to_message_content(
      self,
      provider: types.ProviderNameType,
      metadata: message_content.FileUploadMetadata,
  ) -> message_content.MessageContent:
    metadata.provider = provider
    kwargs = {}
    if metadata.mime_type:
      kwargs['media_type'] = metadata.mime_type
    else:
      kwargs['type'] = message_content.ContentType.DOCUMENT
    if metadata.uri:
      kwargs['source'] = metadata.uri
    return message_content.MessageContent(
        filename=metadata.filename,
        provider_file_api_ids={provider: metadata.file_id},
        provider_file_api_status={provider: metadata},
        **kwargs,
    )

  def _list_from_proxdash(self,
                          limit: int) -> list[message_content.MessageContent]:
    if not self._proxdash_connected():
      return []
    try:
      return self.proxdash_connection.list_files(limit=limit)
    except Exception:
      return []

  def _filter_proxdash_by_providers(
      self,
      proxdash_results: list[message_content.MessageContent],
      providers: list[types.ProviderNameType],
  ) -> list[message_content.MessageContent]:
    provider_set = set(providers)
    filtered = []
    for mc in proxdash_results:
      if not mc.provider_file_api_ids:
        continue
      if provider_set.intersection(mc.provider_file_api_ids):
        filtered.append(mc)
    return filtered

  def _build_covered_file_ids(
      self, proxdash_results: list[message_content.MessageContent]
  ) -> dict[str, set[str]]:
    covered: dict[str, set[str]] = {}
    for mc in proxdash_results:
      if mc.provider_file_api_ids is None:
        continue
      for provider, file_id in mc.provider_file_api_ids.items():
        if provider not in covered:
          covered[provider] = set()
        covered[provider].add(file_id)
    return covered

  def _execute_lists(
      self,
      providers: list[types.ProviderNameType],
      limit: int,
  ) -> list[message_content.MessageContent]:
    """List from ProxDash and providers in parallel, deduplicate."""
    include_proxdash = self._proxdash_connected()
    use_parallel = self._use_parallel(providers) or include_proxdash

    provider_results_raw: dict[str,
                               list[message_content.FileUploadMetadata]] = {}
    proxdash_results: list[message_content.MessageContent] = []

    if use_parallel:
      max_workers = len(providers) + (1 if include_proxdash else 0)
      with ThreadPoolExecutor(max_workers=max_workers) as pool:
        provider_futures = {
            provider: pool.submit(self._list_single_provider, provider, limit)
            for provider in providers
        }
        proxdash_future = None
        if include_proxdash:
          proxdash_future = pool.submit(self._list_from_proxdash, limit)

        for provider, future in provider_futures.items():
          try:
            provider_results_raw[provider] = future.result()
          except Exception:
            provider_results_raw[provider] = []

        if proxdash_future is not None:
          try:
            proxdash_results = proxdash_future.result()
          except Exception:
            proxdash_results = []
    else:
      for provider in providers:
        try:
          provider_results_raw[provider] = (
              self._list_single_provider(provider, limit)
          )
        except Exception:
          provider_results_raw[provider] = []

    proxdash_results = self._filter_proxdash_by_providers(
        proxdash_results, providers
    )
    covered = self._build_covered_file_ids(proxdash_results)

    provider_results: list[message_content.MessageContent] = []
    for provider, metadatas in provider_results_raw.items():
      provider_covered = covered.get(provider, set())
      for meta in metadatas:
        if meta.file_id in provider_covered:
          continue
        provider_results.append(
            self._metadata_to_message_content(provider, meta)
        )

    return proxdash_results + provider_results

  def list(
      self,
      providers: list[types.ProviderNameType] | None = None,
      limit_per_provider: int = 100,
  ) -> list[message_content.MessageContent]:
    """List files from ProxDash and provider File APIs.

    When ProxDash is connected, fetches files from ProxDash first
    (which contain combined provider metadata), then fetches from
    individual providers and skips files already covered by ProxDash.

    Args:
      providers: List of provider names to query, or None to query
        all providers with API keys configured.
      limit_per_provider: Maximum number of files to fetch from each
        provider. Defaults to 100.

    Returns:
      A list of MessageContent objects. ProxDash results come first
      (with combined provider metadata), followed by provider-only
      files not tracked by ProxDash.

    Raises:
      ValueError: If providers is an empty list or no providers
        have API keys configured.
    """
    providers = self._resolve_list_providers(providers)
    for provider in providers:
      self._validate_provider_support(provider, self._get_list_dispatch())

    return self._execute_lists(providers, limit_per_provider)

  # --- Remove ---

  def _resolve_remove_providers(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType] | None,
  ) -> list[types.ProviderNameType]:
    if providers is not None and len(providers) == 0:
      raise ValueError(
          "'providers' must contain at least one provider name, "
          "or be omitted to remove from all uploaded providers."
      )
    if providers is None:
      if (
          media.provider_file_api_ids is None or not media.provider_file_api_ids
      ):
        if self._proxdash_connected() and media.proxdash_file_id is not None:
          return []
        raise ValueError("No uploaded providers found on this media content.")
      return list(media.provider_file_api_ids.keys())
    return providers

  def _validate_remove_providers(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
  ):
    for provider in providers:
      self._validate_provider_support(provider, self._get_remove_dispatch())
      if (
          media.provider_file_api_ids is None or
          provider not in media.provider_file_api_ids
      ):
        raise ValueError(
            f"No file_id found for provider '{provider}' "
            f"on this media content."
        )

  def _remove_single_provider(
      self,
      provider: types.ProviderNameType,
      file_id: str,
  ):
    token_map = self.api_key_manager.get_provider_keys(provider)
    remove_fn = self._get_remove_dispatch()[provider]
    remove_fn(file_id=file_id, token_map=token_map)

  def _collect_remove_result(
      self,
      media: message_content.MessageContent,
      provider: types.ProviderNameType,
  ):
    del media.provider_file_api_status[provider]
    del media.provider_file_api_ids[provider]

  def _execute_removes(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
  ) -> dict[str, Exception]:
    """Remove from all providers and ProxDash in parallel."""
    errors: dict[str, Exception] = {}
    include_proxdash = (
        self._proxdash_connected() and media.proxdash_file_id is not None
    )
    use_parallel = self._use_parallel(providers) or include_proxdash

    if use_parallel:
      max_workers = len(providers) + (1 if include_proxdash else 0)
      with ThreadPoolExecutor(max_workers=max_workers) as pool:
        provider_futures = {
            provider:
                pool.submit(
                    self._remove_single_provider,
                    provider,
                    media.provider_file_api_ids[provider],
                ) for provider in providers
        }
        proxdash_future = None
        if include_proxdash:
          proxdash_future = pool.submit(
              self.proxdash_connection.delete_file, media.proxdash_file_id
          )

        for provider, future in provider_futures.items():
          try:
            future.result()
            self._collect_remove_result(media, provider)
          except Exception as e:
            errors[provider] = e

        if proxdash_future is not None:
          try:
            proxdash_future.result()
          except Exception:
            pass
          media.proxdash_file_id = None
          media.proxdash_file_status = None
    else:
      for provider in providers:
        try:
          self._remove_single_provider(
              provider, media.provider_file_api_ids[provider]
          )
          self._collect_remove_result(media, provider)
        except Exception as e:
          errors[provider] = e
    return errors

  def remove(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType] | None = None,
  ) -> message_content.MessageContent:
    """Remove uploaded files from provider File APIs and ProxDash.

    When ProxDash is connected and the media has a proxdash_file_id,
    also deletes the file from ProxDash. ProxDash deletion failure
    is silent and does not affect provider removals.

    Args:
      media: A MessageContent with provider_file_api_ids populated
        from a previous upload.
      providers: List of provider names to remove from, or None to
        remove from all uploaded providers.

    Returns:
      The same MessageContent with removed providers cleared from
      provider_file_api_status and provider_file_api_ids, and
      proxdash fields cleared.

    Raises:
      ValueError: If providers is an empty list, or if media has no
        uploaded files.
      FileRemoveError: If one or more provider removals fail.
        Successfully removed providers are cleared from the media
        object; failed providers remain.
    """
    providers = self._resolve_remove_providers(media, providers)
    self._validate_remove_providers(media, providers)

    errors = self._execute_removes(media, providers)
    if errors:
      raise FileRemoveError(errors=errors, media=media)

    return media

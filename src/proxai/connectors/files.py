"""File management for provider File APIs."""

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import proxai.chat.message_content as message_content
import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.proxdash as proxdash
import proxai.connectors.file_upload_helpers as file_upload_helpers
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_FILES_MANAGER_STATE_PROPERTY = '_files_manager_state'


class FileUploadError(Exception):
  """Raised when one or more provider uploads fail."""

  def __init__(
      self, errors: dict[str, Exception],
      media: message_content.MessageContent
  ):
    self.errors = errors
    self.media = media
    providers = ', '.join(errors.keys())
    super().__init__(f"Upload failed for providers: {providers}")


class FileRemoveError(Exception):
  """Raised when one or more provider file removals fail."""

  def __init__(
      self, errors: dict[str, Exception],
      media: message_content.MessageContent
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
        len(providers) > 1
        and self.provider_call_options is not None
        and self.provider_call_options.allow_parallel_file_operations
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
      raise ValueError(
          f"No API key configured for provider '{provider}'."
      )

  # --- Upload ---

  def _validate_upload_media(
      self, media: message_content.MessageContent
  ):
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
    upload_fn = file_upload_helpers.UPLOAD_DISPATCH[provider]
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

  def _execute_uploads(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
      file_path: str | None,
      file_data: bytes | None,
      filename: str,
      mime_type: str,
  ) -> dict[str, Exception]:
    errors: dict[str, Exception] = {}
    if self._use_parallel(providers):
      with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {
            provider: pool.submit(
                self._upload_single_provider,
                provider, file_path, file_data, filename, mime_type,
            )
            for provider in providers
        }
        for provider, future in futures.items():
          try:
            metadata = future.result()
            self._collect_upload_result(media, provider, metadata)
          except Exception as e:
            self._collect_upload_error(media, provider, errors, e)
    else:
      for provider in providers:
        try:
          metadata = self._upload_single_provider(
              provider, file_path, file_data, filename, mime_type)
          self._collect_upload_result(media, provider, metadata)
        except Exception as e:
          self._collect_upload_error(media, provider, errors, e)
    return errors

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
    for provider in providers:
      self._validate_provider_support(
          provider, file_upload_helpers.UPLOAD_DISPATCH)

    file_path, file_data, filename, mime_type = (
        self._resolve_upload_file_info(media))
    self._init_upload_fields(media)

    errors = self._execute_uploads(
        media, providers, file_path, file_data, filename, mime_type)
    if errors:
      raise FileUploadError(errors=errors, media=media)

    return media

  # --- Download / List ---

  def download(self):
    pass

  def list(self):
    pass

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
      if (media.provider_file_api_ids is None
          or not media.provider_file_api_ids):
        raise ValueError(
            "No uploaded providers found on this media content."
        )
      return list(media.provider_file_api_ids.keys())
    return providers

  def _validate_remove_providers(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType],
  ):
    for provider in providers:
      self._validate_provider_support(
          provider, file_upload_helpers.REMOVE_DISPATCH)
      if (media.provider_file_api_ids is None
          or provider not in media.provider_file_api_ids):
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
    remove_fn = file_upload_helpers.REMOVE_DISPATCH[provider]
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
    errors: dict[str, Exception] = {}
    if self._use_parallel(providers):
      with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {
            provider: pool.submit(
                self._remove_single_provider,
                provider, media.provider_file_api_ids[provider],
            )
            for provider in providers
        }
        for provider, future in futures.items():
          try:
            future.result()
            self._collect_remove_result(media, provider)
          except Exception as e:
            errors[provider] = e
    else:
      for provider in providers:
        try:
          self._remove_single_provider(
              provider, media.provider_file_api_ids[provider])
          self._collect_remove_result(media, provider)
        except Exception as e:
          errors[provider] = e
    return errors

  def remove(
      self,
      media: message_content.MessageContent,
      providers: list[types.ProviderNameType] | None = None,
  ) -> message_content.MessageContent:
    """Remove uploaded files from provider File APIs.

    Args:
      media: A MessageContent with provider_file_api_ids populated
        from a previous upload.
      providers: List of provider names to remove from, or None to
        remove from all uploaded providers.

    Returns:
      The same MessageContent with removed providers cleared from
      provider_file_api_status and provider_file_api_ids.

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

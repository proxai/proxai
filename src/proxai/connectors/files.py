"""File management for provider File APIs."""

import dataclasses
import os

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
    _MEDIA_TYPES = (
        message_content.ContentType.IMAGE,
        message_content.ContentType.DOCUMENT,
        message_content.ContentType.AUDIO,
        message_content.ContentType.VIDEO,
    )
    if media.type not in _MEDIA_TYPES:
      raise ValueError(
          f"upload() requires a media content type "
          f"(IMAGE, DOCUMENT, AUDIO, VIDEO), got '{media.type.value}'."
      )
    if media.path is None and media.data is None:
      raise ValueError(
          "MessageContent must have 'path' or 'data' set for upload."
      )

    file_path = media.path
    file_data = media.data
    filename = 'file'
    if file_path:
      filename = os.path.basename(file_path)
    mime_type = media.media_type or 'application/octet-stream'

    if media.provider_file_api_status is None:
      media.provider_file_api_status = {}
    if media.provider_file_api_ids is None:
      media.provider_file_api_ids = {}

    errors: dict[str, Exception] = {}
    for provider in providers:
      if provider not in file_upload_helpers.UPLOAD_DISPATCH:
        raise ValueError(
            f"Provider '{provider}' does not support the File API. "
            f"Supported: {list(file_upload_helpers.UPLOAD_DISPATCH.keys())}"
        )
      if not self.api_key_manager.has_provider_key(provider):
        raise ValueError(
            f"No API key configured for provider '{provider}'."
        )
      token_map = self.api_key_manager.get_provider_keys(provider)
      upload_fn = file_upload_helpers.UPLOAD_DISPATCH[provider]
      try:
        metadata = upload_fn(
            file_path=file_path,
            file_data=file_data,
            filename=filename,
            mime_type=mime_type,
            token_map=token_map,
        )
        media.provider_file_api_status[provider] = metadata
        media.provider_file_api_ids[provider] = metadata.file_id
      except Exception as e:
        media.provider_file_api_status[provider] = (
            message_content.FileUploadMetadata(
                file_id='',
                state=message_content.FileUploadState.FAILED,
            )
        )
        errors[provider] = e

    if errors:
      raise FileUploadError(errors=errors, media=media)

    return media

  def download(self):
    pass

  def list(self):
    pass

  def remove(self):
    pass

import copy
import dataclasses
import json
import os
from importlib.metadata import version
from typing import Union

import requests

import proxai.experiment.experiment as experiment
import proxai.logging.utils as logging_utils
import proxai.serializers.type_serializer as type_serializer
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_PROXDASH_STATE_PROPERTY = '_proxdash_connection_state'
_NOT_SET_EXPERIMENT_PATH_VALUE = '(not set)'
_SENSITIVE_CONTENT_HIDDEN_STRING = '<SENSITIVE CONTENT HIDDEN>'


@dataclasses.dataclass
class ProxDashConnectionParams:
  """Initialization parameters for ProxDashConnection."""

  hidden_run_key: str | None = None
  experiment_path: str | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_options: types.ProxDashOptions | None = None


class ProxDashConnection(state_controller.StateControlled):
  """Manages connection and data upload to the ProxDash service."""

  _UPLOADABLE_MEDIA_TYPES = (
      types.ContentType.IMAGE,
      types.ContentType.DOCUMENT,
      types.ContentType.AUDIO,
      types.ContentType.VIDEO,
  )

  _status: types.ProxDashConnectionStatus | None
  _hidden_run_key: str | None
  _experiment_path: str | None
  _logging_options: types.LoggingOptions | None
  _proxdash_options: types.ProxDashOptions | None
  _key_info_from_proxdash: dict | None
  _connected_experiment_path: str | None
  _proxdash_connection_state: types.ProxDashConnectionState | None

  def __init__(
      self, init_from_params: ProxDashConnectionParams | None = None,
      init_from_state: types.ProxDashConnectionState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    if init_from_state:
      self.load_state(init_from_state)
    else:
      self.status = types.ProxDashConnectionStatus.INITIALIZING

      self.hidden_run_key = init_from_params.hidden_run_key
      self.logging_options = init_from_params.logging_options
      self.proxdash_options = init_from_params.proxdash_options
      self.experiment_path = init_from_params.experiment_path

      self._init_connection()

  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    return _PROXDASH_STATE_PROPERTY

  def get_internal_state_type(cls):
    """Return the dataclass type used for state storage."""
    return types.ProxDashConnectionState

  @property
  def hidden_run_key(self) -> str | None:
    return self.get_property_value('hidden_run_key')

  @hidden_run_key.setter
  def hidden_run_key(self, hidden_run_key: str | None):
    self.set_property_value('hidden_run_key', hidden_run_key)

  @property
  def logging_options(self) -> types.LoggingOptions:
    if self._logging_options is None:
      self._logging_options = types.LoggingOptions()
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self.set_property_value('logging_options', logging_options)

  @property
  def proxdash_options(self) -> types.ProxDashOptions:
    if self._proxdash_options is None:
      self._proxdash_options = types.ProxDashOptions()
    return self.get_property_value('proxdash_options')

  @proxdash_options.setter
  def proxdash_options(self, proxdash_options: types.ProxDashOptions):
    self.set_property_value('proxdash_options', proxdash_options)

  @property
  def key_info_from_proxdash(self) -> dict | None:
    return self.get_property_value('key_info_from_proxdash')

  @key_info_from_proxdash.setter
  def key_info_from_proxdash(self, key_info_from_proxdash: dict | None):
    self.set_property_value('key_info_from_proxdash', key_info_from_proxdash)

  @property
  def experiment_path(self) -> str:
    internal_experiment_path = self.get_property_internal_value(
        'experiment_path'
    )

    experiment_path = None
    if (
        internal_experiment_path is not None and
        internal_experiment_path != _NOT_SET_EXPERIMENT_PATH_VALUE
    ):
      experiment_path = internal_experiment_path

    if experiment_path is None:
      experiment_path = _NOT_SET_EXPERIMENT_PATH_VALUE

    self.set_property_internal_state_value('experiment_path', experiment_path)
    return experiment_path

  @experiment_path.setter
  def experiment_path(self, experiment_path: str | None):
    if experiment_path is not None:
      experiment.validate_experiment_path(experiment_path)
    else:
      experiment_path = _NOT_SET_EXPERIMENT_PATH_VALUE

    self.set_property_value('experiment_path', experiment_path)

  @property
  def connected_experiment_path(self) -> str:
    return self.get_property_value('connected_experiment_path')

  @connected_experiment_path.setter
  def connected_experiment_path(self, connected_experiment_path: str | None):
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      if connected_experiment_path is not None:
        raise ValueError(
            'Connected experiment path can only be set if the ProxDash '
            'connection is connected.'
        )
      self.set_property_value_without_triggering_getters(
          'connected_experiment_path', None
      )
      return

    previous_experiment_path = self.get_property_internal_value(
        'connected_experiment_path'
    )
    if previous_experiment_path is None:
      previous_experiment_path = _NOT_SET_EXPERIMENT_PATH_VALUE

    new_experiment_path = connected_experiment_path
    if new_experiment_path is None:
      new_experiment_path = _NOT_SET_EXPERIMENT_PATH_VALUE

    if previous_experiment_path != new_experiment_path:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Connected to ProxDash experiment: '
              f'{new_experiment_path}'
          ), type=types.LoggingType.INFO
      )

    self.set_property_value(
        'connected_experiment_path', connected_experiment_path
    )

  @property
  def status(self) -> types.ProxDashConnectionStatus:
    return self.get_property_value('status')

  @status.setter
  def status(self, status: types.ProxDashConnectionStatus):
    self.set_property_value('status', status)
    if status == types.ProxDashConnectionStatus.INITIALIZING:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message='ProxDash connection initializing.',
          type=types.LoggingType.INFO
      )
    elif status == types.ProxDashConnectionStatus.DISABLED:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message='ProxDash connection disabled.', type=types.LoggingType.INFO
      )
    elif status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash connection disabled. Please provide a valid API key '
              'either as an argument or as an environment variable.'
          ), type=types.LoggingType.ERROR
      )
    elif status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash API key not valid. Please provide a valid API key.\n'
              'Check proxai.co/dashboard/api-keys page to get your API '
              'key.'
          ), type=types.LoggingType.ERROR
      )
    elif status == types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash returned an invalid response.\nPlease report this '
              'issue to the https://github.com/proxai/proxai.\n'
              'Also, please check latest stable version of ProxAI.'
          ), type=types.LoggingType.ERROR
      )
    elif status == types.ProxDashConnectionStatus.CONNECTED:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              f'Connected to ProxDash at {self.proxdash_options.base_url}'
          ), type=types.LoggingType.INFO
      )

  def _init_connection(self):
    proxdash_disabled = self.proxdash_options.disable_proxdash
    if proxdash_disabled:
      self.status = types.ProxDashConnectionStatus.DISABLED
      self.key_info_from_proxdash = None
      # Note: There is no longer any connection to ProxDash. This change
      # shouldn't be logged, so, self.connected_experiment_path setter should
      # not be used here.
      self.set_property_value_without_triggering_getters(
          'connected_experiment_path', None
      )
      return

    if self.proxdash_options.api_key is None:
      if 'PROXDASH_API_KEY' not in os.environ:
        self.status = types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
        self.key_info_from_proxdash = None
        # Note: There is no longer any connection to ProxDash. This change
        # shouldn't be logged, so, self.connected_experiment_path setter should
        # not be used here.
        self.set_property_value_without_triggering_getters(
            'connected_experiment_path', None
        )
        return
      else:
        # Note: Setting api_key from environment variable.
        self.proxdash_options.api_key = os.environ['PROXDASH_API_KEY']

    validation_status, key_info_from_proxdash = self._check_api_key_validity(
        base_url=self.proxdash_options.base_url,
        api_key=self.proxdash_options.api_key
    )
    self.status = validation_status
    self.key_info_from_proxdash = key_info_from_proxdash

    if self.status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID:
      self.key_info_from_proxdash = None
      self.set_property_value_without_triggering_getters(
          'connected_experiment_path', None
      )
      raise ValueError(
          'ProxDash API key not valid. Please provide a valid API key.\n'
          f'base_url: {self.proxdash_options.base_url}\n'
          f'api_key: {self.proxdash_options.api_key[:3]}...\n\n'
          'To fix this issue:\n'
          '1. Check that your PROXDASH_API_KEY in your .bashrc or .zshrc file '
          'is correct if it exists\n'
          '2. Check that px.ProxDashOptions(api_key="your_api_key") is correct '
          'if you set it directly\n'
          '3. Verify that your key matches what appears on '
          'https://proxai.co/dashboard/api-keys\n'
          '4. If you don\'t want to use ProxDash, make sure PROXDASH_API_KEY '
          'is not set in your environment variables\n'
          'For more information, see: '
          'https://www.proxai.co/proxai-docs/advanced/proxdash-connection'
      )

    if self.status == types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN:
      self.key_info_from_proxdash = None
      self.set_property_value_without_triggering_getters(
          'connected_experiment_path', None
      )
      raise ValueError(
          'ProxDash returned an invalid response.\nPlease report this '
          'issue to the https://github.com/proxai/proxai.\n'
          'Also, please check latest stable version of ProxAI.'
      )

    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      self.set_property_value_without_triggering_getters(
          'connected_experiment_path', None
      )
      raise ValueError(
          'Unknown ProxDash connection status.\n'
          f'self.status: {self.status}\n'
          'result_state.key_info_from_proxdash: '
          f'{self.key_info_from_proxdash}'
      )

    if self.connected_experiment_path != self.experiment_path:
      self.connected_experiment_path = self.experiment_path

  def _check_api_key_validity(
      self,
      base_url: str,
      api_key: str,
  ) -> tuple[
      Union[  # noqa: UP007
          types.ProxDashConnectionStatus.API_KEY_NOT_VALID,
          types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN,
          types.ProxDashConnectionStatus.CONNECTED,
      ],
      dict | None,
  ]:
    response = requests.get(
        f'{base_url}/ingestion/verify-key', headers={'X-API-Key': api_key}
    )
    if response.status_code != 200 or response.text == 'false':
      return types.ProxDashConnectionStatus.API_KEY_NOT_VALID, None
    try:
      api_response = json.loads(response.text)
      # New backend API response format
      if api_response.get('success') and api_response.get('data'):
        return types.ProxDashConnectionStatus.CONNECTED, api_response['data']
      # Old backend API response format
      if 'keyName' in api_response:
        return types.ProxDashConnectionStatus.CONNECTED, api_response
      return types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN, None
    except Exception:
      return types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN, None

  def _hide_sensitive_content(
      self, call_record: types.CallRecord
  ) -> types.CallRecord:
    call_record = copy.deepcopy(call_record)
    if call_record.query:
      if call_record.query.prompt:
        call_record.query.prompt = _SENSITIVE_CONTENT_HIDDEN_STRING
      if call_record.query.system_prompt:
        call_record.query.system_prompt = _SENSITIVE_CONTENT_HIDDEN_STRING
      if call_record.query.chat:
        call_record.query.chat = None
    if call_record.result:
      if call_record.result.output_text:
        call_record.result.output_text = _SENSITIVE_CONTENT_HIDDEN_STRING
      call_record.result.output_image = None
      call_record.result.output_audio = None
      call_record.result.output_video = None
      call_record.result.output_json = None
      call_record.result.output_pydantic = None
      call_record.result.content = None
      call_record.result.choices = None
    return call_record

  def upload_call_record(self, call_record: types.CallRecord):
    """Upload a call record to ProxDash."""
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return
    if ((
        self.proxdash_options and self.proxdash_options.hide_sensitive_content
    ) or self._key_info_from_proxdash['permission'] == 'NO_PROMPT'):
      call_record = self._hide_sensitive_content(call_record)

    data = type_serializer.encode_call_record(call_record)
    data['schema_version'] = 2
    data['experiment_path'] = self.experiment_path
    data.setdefault('connection', {})['caller_type'] = 'SDK'
    data.setdefault('connection', {})['caller_app'] = 'PYTHON_SDK'
    if 'result' in data and 'role' in data['result']:
      data['result']['role'] = data['result']['role'].upper()

    try:
      response = requests.post(
          f'{self.proxdash_options.base_url}/ingestion/call-records', json=data,
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=('ProxDash could not log the record. Error: '
                   f'{str(e)}'), type=types.LoggingType.ERROR
      )
      return

    if response.status_code != 201:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash could not log the record. Status code: '
              f'{response.status_code}, Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return

    try:
      api_response = json.loads(response.text)
    except Exception:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash could not log the record. Invalid JSON response: '
              f'{response.text}'
          ), type=types.LoggingType.ERROR
      )
      return

    if not api_response.get('success'):
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash could not log the record. Response: '
              f'{response.text}'
          ), type=types.LoggingType.ERROR
      )

  def _resolve_file_bytes(self, media: types.MessageContent) -> bytes | None:
    if media.data is not None:
      return media.data
    if media.path is not None:
      with open(media.path, 'rb') as f:
        return f.read()
    return None

  def _request_presigned_upload_url(
      self, filename: str, mime_type: str, size_bytes: int
  ) -> tuple[str, str, str] | None:
    """Returns (file_id, presigned_url, s3_key) or None on failure."""
    try:
      resp = requests.post(
          f'{self.proxdash_options.base_url}/files/upload', json={
              'filename': filename,
              'mimeType': mime_type,
              'sizeBytes': size_bytes,
          }, headers={'X-API-Key': self.proxdash_options.api_key}
      )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'ProxDash file upload request failed. Error: {e}',
          type=types.LoggingType.ERROR
      )
      return None

    if resp.status_code != 201:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash file upload request failed. '
              f'Status: {resp.status_code}, Response: {resp.text}'
          ), type=types.LoggingType.ERROR
      )
      return None

    try:
      upload_result = json.loads(resp.text)
      if not upload_result.get('success'):
        logging_utils.log_proxdash_message(
            logging_options=self.logging_options,
            proxdash_options=self.proxdash_options, message=(
                'ProxDash file upload request failed. '
                f'Response: {resp.text}'
            ), type=types.LoggingType.ERROR
        )
        return None
      data = upload_result['data']
      return data['id'], data['presignedUploadUrl'], data['s3Key']
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'ProxDash file upload response parse failed. '
              f'Error: {e}'
          ), type=types.LoggingType.ERROR
      )
      return None

  def _put_file_to_s3(
      self, presigned_url: str, file_bytes: bytes, mime_type: str
  ) -> bool:
    try:
      resp = requests.put(
          presigned_url, data=file_bytes, headers={'Content-Type': mime_type}
      )
      if resp.status_code not in (200, 204):
        logging_utils.log_proxdash_message(
            logging_options=self.logging_options,
            proxdash_options=self.proxdash_options, message=(
                'ProxDash S3 upload failed. '
                f'Status: {resp.status_code}'
            ), type=types.LoggingType.ERROR
        )
        return False
      return True
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'ProxDash S3 upload failed. Error: {e}',
          type=types.LoggingType.ERROR
      )
      return False

  def _confirm_file_upload(self, file_id: str):
    try:
      resp = requests.post(
          f'{self.proxdash_options.base_url}/files/update/{file_id}',
          json={'uploadConfirmed': True},
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
      if resp.status_code != 200:
        logging_utils.log_proxdash_message(
            logging_options=self.logging_options,
            proxdash_options=self.proxdash_options, message=(
                'ProxDash file confirm failed. '
                f'Status: {resp.status_code}, '
                f'Response: {resp.text}'
            ), type=types.LoggingType.ERROR
        )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'ProxDash file confirm failed. Error: {e}',
          type=types.LoggingType.ERROR
      )

  def upload_file(self, media: types.MessageContent) -> str | None:
    """Upload a file to ProxDash via presigned S3 URL.

    Returns the ProxDash file ID on success, None on failure.
    Sets media.proxdash_file_id and media.proxdash_file_status.
    """
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return None
    if media.type not in self._UPLOADABLE_MEDIA_TYPES:
      return None

    file_bytes = self._resolve_file_bytes(media)
    if file_bytes is None:
      return None

    filename = (
        media.filename or
        (os.path.basename(media.path) if media.path else None) or
        '(no_filename_set)'
    )
    mime_type = media.media_type or 'application/octet-stream'

    upload_info = self._request_presigned_upload_url(
        filename, mime_type, len(file_bytes)
    )
    if upload_info is None:
      return None
    file_id, presigned_url, s3_key = upload_info

    if not self._put_file_to_s3(presigned_url, file_bytes, mime_type):
      return None

    self._confirm_file_upload(file_id)

    media.proxdash_file_id = file_id
    media.proxdash_file_status = types.ProxDashFileStatus(
        file_id=file_id,
        s3_key=s3_key,
        upload_confirmed=True,
    )
    return file_id

  def update_file(self, media: types.MessageContent):
    """Update a ProxDash file record with provider metadata."""
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return
    if media.proxdash_file_id is None:
      return
    update_data = {}
    if media.provider_file_api_ids:
      update_data['providerFileApiIds'] = media.provider_file_api_ids
    if media.provider_file_api_status:
      update_data['providerFileApiStatus'] = {
          prov: meta.to_dict()
          for prov, meta in media.provider_file_api_status.items()
      }
    if not update_data:
      return
    file_id = media.proxdash_file_id

    try:
      resp = requests.post(
          (f'{self.proxdash_options.base_url}'
           f'/files/update/{file_id}'), json=update_data,
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
      if resp.status_code != 200:
        logging_utils.log_proxdash_message(
            logging_options=self.logging_options,
            proxdash_options=self.proxdash_options, message=(
                'ProxDash file update failed. '
                f'Status: {resp.status_code}, '
                f'Response: {resp.text}'
            ), type=types.LoggingType.ERROR
        )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'ProxDash file update failed. Error: {e}',
          type=types.LoggingType.ERROR
      )

  def _file_info_to_message_content(
      self, info: dict
  ) -> types.MessageContent | None:
    """Convert a ProxDash FileInfo response to a MessageContent.

    Maps camelCase backend keys to the snake_case dict that
    MessageContent.from_dict() expects.
    """
    try:
      data = {
          'type': info.get('type', 'document'),
          'filename': info.get('filename'),
          'provider_file_api_ids': info.get('providerFileApiIds'),
          'provider_file_api_status': info.get('providerFileApiStatus'),
          'proxdash_file_id': info.get('id'),
          'proxdash_file_status': {
              'file_id': info.get('id', ''),
              'upload_confirmed': info.get('uploadConfirmed', False),
              'source': info.get('source'),
              'created_at': (
                  str(info['createdAt'])
                  if info.get('createdAt') else None
              ),
              'updated_at': (
                  str(info['updatedAt'])
                  if info.get('updatedAt') else None
              ),
          },
      }
      if info.get('mimeType'):
        data['media_type'] = info['mimeType']
      if info.get('source'):
        data['source'] = info['source']
      return types.MessageContent.from_dict(data)
    except Exception:
      return None

  def get_file(
      self, file_id: str
  ) -> types.MessageContent | None:
    """Get file info from ProxDash as a MessageContent."""
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return None
    try:
      resp = requests.get(
          f'{self.proxdash_options.base_url}/files/{file_id}',
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
      if resp.status_code != 200:
        return None
      result = json.loads(resp.text)
      if result.get('success') and result.get('data'):
        return self._file_info_to_message_content(result['data'])
      return None
    except Exception:
      return None

  def delete_file(self, file_id: str) -> bool:
    """Delete a file from ProxDash (S3 + DB record).

    Returns True on success, False on failure.
    """
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return False
    try:
      resp = requests.delete(
          f'{self.proxdash_options.base_url}/files/{file_id}',
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
      return resp.status_code == 200
    except Exception:
      return False

  def download_file(self, file_id: str) -> bytes | None:
    """Download file bytes from ProxDash via presigned S3 URL.

    Returns the file bytes on success, None on failure.
    """
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return None
    try:
      resp = requests.get(
          f'{self.proxdash_options.base_url}/files/download/{file_id}',
          headers={'X-API-Key': self.proxdash_options.api_key}
      )
      if resp.status_code != 200:
        return None
      result = json.loads(resp.text)
      if not result.get('success') or not result.get('data'):
        return None
      presigned_url = result['data'].get('url')
      if not presigned_url:
        return None
    except Exception:
      return None

    try:
      download_resp = requests.get(presigned_url)
      if download_resp.status_code != 200:
        return None
      return download_resp.content
    except Exception:
      return None

  def list_files(
      self, limit: int = 100
  ) -> list[types.MessageContent]:
    """List files from ProxDash as MessageContent objects.

    Returns a list of MessageContent with proxdash_file_id,
    proxdash_file_status, and any synced provider metadata.
    Empty list on failure.
    """
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return []
    try:
      resp = requests.get(
          f'{self.proxdash_options.base_url}/files/list', params={
              'pageSize': limit,
              'includeSource': 'true',
          }, headers={'X-API-Key': self.proxdash_options.api_key}
      )
      if resp.status_code != 200:
        return []
      result = json.loads(resp.text)
      if not result.get('success') or not result.get('data'):
        return []
      results = []
      for info in result['data'].get('items', []):
        mc = self._file_info_to_message_content(info)
        if mc is not None:
          results.append(mc)
      return results
    except Exception:
      return []

  def get_model_configs_schema(self,) -> types.ModelConfigsSchemaType | None:
    """Fetch the latest model configurations from ProxDash."""
    current_version = version("proxai")
    request_url = (
        f'{self.proxdash_options.base_url}' +
        f'/models/configs?proxaiVersion={current_version}'
    )
    if self.status == types.ProxDashConnectionStatus.CONNECTED:
      response = requests.get(
          request_url, headers={'X-API-Key': self.proxdash_options.api_key}
      )
    else:
      response = requests.get(request_url)

    if response.status_code != 200:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Failed to get model configs from ProxDash.\n'
              f'ProxAI version: {current_version}\n'
              f'Status code: {response.status_code}\n'
              f'Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return None

    response_data = json.loads(response.text)
    if not response_data['success']:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Failed to get model configs from ProxDash.\n'
              f'ProxAI version: {current_version}\n'
              f'Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return None

    try:
      model_configs_schema = type_serializer.decode_model_configs_schema_type(
          response_data['data']
      )
    except Exception as e:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Failed to decode model configs from ProxDash response.\n'
              'Please report this issue to the '
              'https://github.com/proxai/proxai.\n'
              'Also, please check latest stable version of ProxAI.\n'
              f'ProxAI version: {current_version}\n'
              f'Error: {str(e)}'
          ), type=types.LoggingType.ERROR
      )
      return None

    if (
        model_configs_schema.metadata is None or
        model_configs_schema.version_config is None or
        model_configs_schema.version_config.provider_model_configs is None or
        len(model_configs_schema.version_config.provider_model_configs) < 2
    ):
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Model configs schema is invalid. Please report this '
              'issue to the https://github.com/proxai/proxai.\n'
              'Also, please check latest stable version of ProxAI. '
              f'Request URL: {request_url}'
              f'Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return None

    logging_utils.log_proxdash_message(
        logging_options=self.logging_options,
        proxdash_options=self.proxdash_options, message=(
            f'Model configs schema (v{model_configs_schema.metadata.version}) '
            'fetched from ProxDash.'
        ), type=types.LoggingType.INFO
    )

    return model_configs_schema

  def get_provider_api_keys(self) -> types.ProviderTokenValueMap:
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return {}

    request_url = (
        f'{self.proxdash_options.base_url}' +
        '/ingestion/provider-connection-keys'
    )
    response = requests.get(
        request_url, headers={'X-API-Key': self.proxdash_options.api_key}
    )

    if response.status_code != 200:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Failed to get provider API keys from ProxDash.\n'
              f'Status code: {response.status_code}\n'
              f'Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return {}

    data = json.loads(response.text)
    if not data['success']:
      logging_utils.log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options, message=(
              'Failed to get provider API keys from ProxDash.\n'
              f'Response: {response.text}'
          ), type=types.LoggingType.ERROR
      )
      return {}

    logging_utils.log_proxdash_message(
        logging_options=self.logging_options,
        proxdash_options=self.proxdash_options,
        message=('Provider API keys fetched from ProxDash.'),
        type=types.LoggingType.INFO
    )

    return data['data']

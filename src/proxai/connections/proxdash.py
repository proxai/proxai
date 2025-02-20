import os
import copy
import json
import requests
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import proxai.experiment.experiment as experiment
from proxai.logging.utils import log_proxdash_message
from typing import Callable, Dict, List, Optional, Union, Tuple

_PROXDASH_BACKEND_URL = 'https://proxainest-production.up.railway.app'


class ProxDashConnection(object):
  _last_connected_experiment_path: Optional[str] = None
  _experiment_path: Optional[str] = None
  _hidden_run_key: str
  _logging_options: types.LoggingOptions
  _get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None
  _status: Optional[types.ProxDashConnectionStatus] = None
  _api_key: str
  _key_info_from_proxdash: Optional[Dict] = None

  def __init__(
      self,
      hidden_run_key: Optional[str] = None,
      api_key: Optional[str] = None,
      experiment_path: Optional[str] = None,
      get_experiment_path: Optional[Callable[[], str]] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None,
      proxdash_options: Optional[types.ProxDashOptions] = None,
      get_proxdash_options: Optional[
          Callable[[], types.ProxDashOptions]] = None,
      init_state: Optional[types.ProxDashInitState] = None):
    if init_state and (
        hidden_run_key or api_key or experiment_path or
        get_experiment_path or logging_options or get_logging_options or
        proxdash_options or get_proxdash_options):
      raise ValueError(
          'If init_state is provided, none of the other arguments should be '
          'provided.')

    init_status = None
    init_key_info_from_proxdash = None
    if init_state:
      init_status = init_state.status
      hidden_run_key = init_state.hidden_run_key
      api_key = init_state.api_key
      experiment_path = init_state.experiment_path
      logging_options = init_state.logging_options
      proxdash_options = init_state.proxdash_options
      init_key_info_from_proxdash = init_state.key_info_from_proxdash

    if experiment_path and get_experiment_path:
      raise ValueError(
          'Only one of experiment_path or get_experiment_path should be '
          'provided.')
    if logging_options and get_logging_options:
      raise ValueError(
          'Only one of logging_options or get_logging_options should be '
          'provided.')
    if proxdash_options and get_proxdash_options:
      raise ValueError(
          'Only one of proxdash_options or get_proxdash_options should be '
          'provided.')

    self.status = types.ProxDashConnectionStatus.INITIALIZING
    self._key_info_from_proxdash = None
    self._hidden_run_key = hidden_run_key
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    self.proxdash_options = proxdash_options
    self._get_proxdash_options = get_proxdash_options
    self.experiment_path = experiment_path
    self._get_experiment_path = get_experiment_path

    self.connect_to_proxdash(
        api_key,
        init_status=init_status,
        init_key_info_from_proxdash=init_key_info_from_proxdash)

  def connect_to_proxdash(
      self,
      api_key: Optional[str] = None,
      init_status: Optional[types.ProxDashConnectionStatus] = None,
      init_key_info_from_proxdash: Optional[Dict] = None
  ):
    if self.proxdash_options and self.proxdash_options.disable_proxdash:
      self.status = types.ProxDashConnectionStatus.DISABLED
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=(
              'ProxDash disabled via proxdash_options. '
              'No data will be sent to ProxDash servers.'),
          type=types.LoggingType.INFO)
      return

    if not api_key:
      if 'PROXDASH_API_KEY' in os.environ:
        self.status = types.ProxDashConnectionStatus.API_KEY_FOUND
        api_key = os.environ['PROXDASH_API_KEY']
      else:
        self.status = types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
        log_proxdash_message(
            logging_options=self.logging_options,
            proxdash_options=self.proxdash_options,
            message=(
                'ProxDash API key not found. Please provide an API key '
                'either as an argument or as an environment variable.'),
            type=types.LoggingType.ERROR)
        return
    self._api_key = api_key

    if init_status == types.ProxDashConnectionStatus.CONNECTED:
      self.status = types.ProxDashConnectionStatus.CONNECTED
      self._key_info_from_proxdash = init_key_info_from_proxdash
      return
    self._check_api_key_validity_and_update_status()

  @property
  def api_key(self) -> str:
    return self._api_key

  @api_key.setter
  def api_key(self, api_key: str):
    self._api_key = api_key
    self._check_api_key_validity_and_update_status()

  @property
  def logging_options(self) -> types.LoggingOptions:
    if self._logging_options:
      return self._logging_options
    if self._get_logging_options:
      return self._get_logging_options()
    return types.LoggingOptions()

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self._logging_options = logging_options

  @property
  def proxdash_options(self) -> types.ProxDashOptions:
    if self._proxdash_options:
      return self._proxdash_options
    if self._get_proxdash_options:
      return self._get_proxdash_options()
    return types.ProxDashOptions()

  @proxdash_options.setter
  def proxdash_options(self, proxdash_options: types.ProxDashOptions):
    self._proxdash_options = proxdash_options

  def _hide_sensitive_content_logging_record(
      self, logging_record: types.LoggingRecord) -> types.LoggingRecord:
    logging_record = copy.deepcopy(logging_record)
    if logging_record.query_record and logging_record.query_record.prompt:
      logging_record.query_record.prompt = '<sensitive content hidden>'
    if logging_record.query_record and logging_record.query_record.system:
      logging_record.query_record.system = '<sensitive content hidden>'
    if logging_record.query_record and logging_record.query_record.messages:
      logging_record.query_record.messages = [
        {
          'role': 'assistant',
          'content': '<sensitive content hidden>'
        }
      ]
    if (logging_record.response_record and
        logging_record.response_record.response):
      logging_record.response_record.response = '<sensitive content hidden>'
    return logging_record

  def _check_api_key_validity(self) -> Tuple[
      Union[
          types.ProxDashConnectionStatus.API_KEY_VALID,
          types.ProxDashConnectionStatus.API_KEY_NOT_VALID,
          types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN],
      Optional[Dict]]:
    response = requests.post(
        f'{_PROXDASH_BACKEND_URL}/connect',
        data={'apiKey': self._api_key})
    if response.status_code != 201 or response.text == 'false':
      return types.ProxDashConnectionStatus.API_KEY_NOT_VALID, None
    try:
      key_info_from_proxdash = json.loads(response.text)
      return (
          types.ProxDashConnectionStatus.API_KEY_VALID,
          key_info_from_proxdash)
    except Exception:
      return (
          types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN,
          None)

  def _check_api_key_validity_and_update_status(self):
    validation_status, key_info_from_proxdash = self._check_api_key_validity()
    if validation_status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID:
      self.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=(
              'ProxDash API key not valid. Please provide a valid API key.\n'
              'Check proxai.co/dashboard/api-keys page to get your API '
              'key.'),
          type=types.LoggingType.ERROR)
    elif (validation_status ==
          types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN):
      self.status = types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=(
              'ProxDash returned an invalid response.\nPlease report this '
              'issue to the https://github.com/proxai/proxai.\n'
              'Also, please check latest stable version of ProxAI.'),
          type=types.LoggingType.ERROR)
    elif validation_status == types.ProxDashConnectionStatus.API_KEY_VALID:
      self.status = types.ProxDashConnectionStatus.CONNECTED
      self._key_info_from_proxdash = key_info_from_proxdash
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message='Connected to ProxDash.',
          type=types.LoggingType.INFO)
      # This will trigger the logging of the connection.
      _ = self.experiment_path

  @property
  def status(self) -> types.ProxDashConnectionStatus:
    return self._status

  @status.setter
  def status(self, status: types.ProxDashConnectionStatus):
    self._status = status

  @property
  def experiment_path(self) -> Optional[str]:
    if self._experiment_path:
      experiment_path = self._experiment_path
    elif self._get_experiment_path:
      experiment_path = self._get_experiment_path()
    else:
      experiment_path = '(not set)'

    if (self._last_connected_experiment_path != experiment_path and
        self.status == types.ProxDashConnectionStatus.CONNECTED):
      self._last_connected_experiment_path = experiment_path
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'Connected to ProxDash experiment: {experiment_path}',
          type=types.LoggingType.INFO)
    return experiment_path

  @experiment_path.setter
  def experiment_path(self, experiment_path: Optional[str]):
    if experiment_path is not None:
      experiment.validate_experiment_path(experiment_path)

    if (self._last_connected_experiment_path != experiment_path and
        self.status == types.ProxDashConnectionStatus.CONNECTED):
      self._last_connected_experiment_path = experiment_path
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=f'Connected to ProxDash experiment: {experiment_path}',
          type=types.LoggingType.INFO)

    self._experiment_path = experiment_path

  def upload_logging_record(self, logging_record: types.LoggingRecord):
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return
    if self.proxdash_options and self.proxdash_options.hide_sensitive_content:
      logging_record = self._hide_sensitive_content_logging_record(
        logging_record)
    stop = None
    if logging_record.query_record.stop:
      stop = str(logging_record.query_record.stop)
    data = {
      'apiKey': self._api_key,
      'hiddenRunKey': self._hidden_run_key,
      'experimentPath': self.experiment_path,
      'callType': logging_record.query_record.call_type,
      'provider': logging_record.query_record.model[0],
      'model': logging_record.query_record.model[1],
      'maxTokens': logging_record.query_record.max_tokens,
      'temperature': logging_record.query_record.temperature,
      'stop': stop,
      'hashValue': logging_record.query_record.hash_value,
      'error': logging_record.response_record.error,
      'errorTraceback': logging_record.response_record.error_traceback,
      'startUTCDate': logging_record.response_record.start_utc_date.isoformat(),
      'endUTCDate': logging_record.response_record.end_utc_date.isoformat(),
      'localTimeOffsetMinute': logging_record.response_record.local_time_offset_minute,
      'responseTime': (
          logging_record.response_record.response_time.total_seconds() * 1000),
      'estimatedCost': logging_record.response_record.estimated_cost,
      'responseSource': logging_record.response_source,
      'lookFailReason': logging_record.look_fail_reason,
    }
    if self._key_info_from_proxdash['permission'] == 'ALL':
      data['prompt'] = logging_record.query_record.prompt
      data['system'] = logging_record.query_record.system
      data['messages'] = logging_record.query_record.messages
      data['response'] = logging_record.response_record.response

    response = requests.post(
        f'{_PROXDASH_BACKEND_URL}/logging-record',
        data=data)

    if response.status_code != 201 or response.text != 'success':
      log_proxdash_message(
          logging_options=self.logging_options,
          proxdash_options=self.proxdash_options,
          message=(
              'ProxDash could not log the record. Error message:\n'
              f'{response.text}'),
          type=types.LoggingType.ERROR)

  def get_init_state(self) -> types.ProxDashInitState:
    return types.ProxDashInitState(
      status=self.status,
      hidden_run_key=self._hidden_run_key,
      api_key=self._api_key,
      experiment_path=self.experiment_path,
      logging_options=self.logging_options,
      proxdash_options=self.proxdash_options,
      key_info_from_proxdash=self._key_info_from_proxdash,
    )

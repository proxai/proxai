import os
import json
import requests
import proxai.types as types
import proxai.serializers.type_serializer as type_serializer
import proxai.experiment.experiment as experiment
from proxai.logging.utils import log_proxdash_message
from typing import Callable, Dict, List, Optional

# _PROXDASH_BACKEND_URL = 'https://proxainest-production.up.railway.app'
_PROXDASH_BACKEND_URL = 'http://localhost:3001'

class ProxDashConnection(object):
  _experiment_name: str = '(not set)'
  _hidden_run_key: str
  _logging_options: types.LoggingOptions
  _get_logging_options: Optional[Callable[[], types.LoggingOptions]]
  _status: types.ProxDashConnectionStatus
  _api_key: str
  _key_data: Optional[Dict]

  def __init__(
      self,
      hidden_run_key: str,
      api_key: Optional[str] = None,
      logging_options: Optional[types.LoggingOptions] = None,
      get_logging_options: Optional[Callable[[], types.LoggingOptions]] = None):
    if logging_options and get_logging_options:
      raise ValueError(
          'Only one of logging_options or get_logging_options should be '
          'provided.')
    self._hidden_run_key = hidden_run_key
    self.logging_options = logging_options
    self._get_logging_options = get_logging_options
    if not api_key:
      if 'PROXDASH_API_KEY' in os.environ:
        api_key = os.environ['PROXDASH_API_KEY']
      else:
        self.status = types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
        log_proxdash_message(
            logging_options=self.logging_options,
            message='ProxDash API key not found. Please provide an API key '
                    'either as an argument or as an environment variable.',
            type=types.LoggingType.ERROR)
        return
    self._api_key = api_key
    if self._check_key_validity():
      self.status = types.ProxDashConnectionStatus.CONNECTED

  @property
  def logging_options(self) -> types.LoggingOptions:
    if self._logging_options:
      return self._logging_options
    if self._get_logging_options:
      return self._get_logging_options()
    return None

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self._logging_options = logging_options

  def _check_key_validity(self):
    response = requests.post(
        f'{_PROXDASH_BACKEND_URL}/connect',
        data={'apiKey': self._api_key})
    if response.status_code != 201 or response.text == 'false':
      self.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID
      log_proxdash_message(
          logging_options=self.logging_options,
          message=(
              'ProxDash API key not valid. Please provide a valid API key.\n'
              'Check proxai.co/dashboard/api-keys page to get your API '
              'key.'),
          type=types.LoggingType.ERROR)
      return False
    try:
      self._key_data = json.loads(response.text)
    except Exception as e:
      self.status = types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
      log_proxdash_message(
          logging_options=self.logging_options,
          message=(
              'ProxDash returned an invalid response.\nPlease report this '
              'issue to the https://github.com/proxai/proxai.\n'
              f'Error message: {e}\nResponse: {response.text}'),
          type=types.LoggingType.ERROR)
      return False
    return True

  @property
  def status(self) -> types.ProxDashConnectionStatus:
    return self._status

  @status.setter
  def status(self, status: types.ProxDashConnectionStatus):
    self._status = status

  @property
  def experiment_name(self) -> str:
    return self._experiment_name

  @experiment_name.setter
  def experiment_name(self, experiment_name) -> str:
    if self._experiment_name == experiment_name:
      return
    experiment.validate_experiment_name(experiment_name)
    self._experiment_name = experiment_name
    if self.status == types.ProxDashConnectionStatus.CONNECTED:
      log_proxdash_message(
          logging_options=self.logging_options,
          message='Connected to ProxDash.\n'
                  f'Experiment name: {experiment_name}',
          type=types.LoggingType.INFO)

  def upload_logging_record(self, logging_record: types.LoggingRecord):
    if self.status != types.ProxDashConnectionStatus.CONNECTED:
      return
    stop = None
    if logging_record.query_record.stop:
      stop = str(logging_record.query_record.stop)
    data = {
      'apiKey': self._api_key,
      'hiddenRunKey': self._hidden_run_key,
      'experimentName': self.experiment_name,
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
    if self._key_data['permission'] == 'ALL':
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
          message=(
              'ProxDash could not log the record. Error message:\n'
              f'{response.text}'),
          type=types.LoggingType.ERROR)

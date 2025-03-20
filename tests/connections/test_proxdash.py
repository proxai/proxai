import os
import copy
import datetime
import json
import tempfile
import pytest
import proxai.types as types
import proxai.connections.proxdash as proxdash
from proxai.logging.utils import log_proxdash_message
import proxai.connectors.model_configs as model_configs
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import requests
from urllib.parse import unquote

@pytest.fixture(autouse=True)
def setup_test(monkeypatch, requests_mock):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  requests_mock.post(
      'https://proxainest-production.up.railway.app/connect',
      text='{"permission": "ALL"}',
      status_code=201,
  )
  yield


def _get_path_dir(temp_path: str):
  temp_dir = tempfile.TemporaryDirectory()
  path = os.path.join(temp_dir.name, temp_path)
  os.makedirs(path, exist_ok=True)
  return path, temp_dir


def _create_test_logging_record(
    prompt: str = "test prompt",
    system: str = "test system",
    messages: Optional[List[Dict[str, str]]] = [],
    response: str = "test response",
    error: Optional[str] = None,
    error_traceback: Optional[str] = None,
    stop: Optional[Union[str, List[str]]] = None,
    hash_value: str = "test_hash"
) -> types.LoggingRecord:
  """Creates a test logging record with default values."""
  if messages == []:
      messages = [
          {"role": "user", "content": "test user message"},
          {"role": "assistant", "content": "test assistant message"}
      ]

  query_record = types.QueryRecord(
      prompt=prompt,
      system=system,
      messages=messages,
      provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
      call_type=types.CallType.GENERATE_TEXT,
      max_tokens=100,
      temperature=0.7,
      stop=stop,
      hash_value=hash_value
  )

  response_record = types.QueryResponseRecord(
      response=response,
      error=error,
      error_traceback=error_traceback,
      start_utc_date=datetime.datetime(2024, 1, 1, 12, 0),
      end_utc_date=datetime.datetime(2024, 1, 1, 12, 1),
      local_time_offset_minute=0,
      response_time=datetime.timedelta(seconds=1),
      estimated_cost=0.0
  )

  return types.LoggingRecord(
      query_record=query_record,
      response_record=response_record,
      response_source=types.ResponseSource.PROVIDER,
      look_fail_reason=None
  )


def _create_connection(
    api_key: str = "test_api_key",
    status: types.ProxDashConnectionStatus = types.ProxDashConnectionStatus.CONNECTED,
    hide_sensitive_content: bool = False,
    permission: str = "ALL",
    temp_dir: Optional[str] = None
) -> Tuple[proxdash.ProxDashConnection, str]:
  """Creates a ProxDashConnection."""
  if temp_dir is None:
      temp_dir, temp_dir_obj = _get_path_dir('test_upload_logging_record')

  connection = proxdash.ProxDashConnection(
      api_key=api_key,
      logging_options=types.LoggingOptions(logging_path=temp_dir),
      proxdash_options=types.ProxDashOptions(
          hide_sensitive_content=hide_sensitive_content),
  )
  connection.status = status
  connection.key_info_from_proxdash = {'permission': permission}

  return connection, temp_dir, temp_dir_obj


def _verify_proxdash_request(
    requests_mock,
    expected_data: Optional[Dict] = None,
    request_id: Optional[int] = None,
    response_status: int = 201,
    response_text: str = 'success'
) -> None:
  """Verifies that the request to ProxDash was made with expected data."""
  logging_record_requests = [
      request for request in requests_mock.request_history
      if request.url ==
      'https://proxainest-production.up.railway.app/logging-record'
  ]
  if expected_data is None:
      assert len(logging_record_requests) == 0
      return

  if request_id is None:
    assert len(logging_record_requests) == 1
    request = logging_record_requests[0]
  else:
    request = logging_record_requests[request_id]
  actual_data = json.loads(request.text)

  def _check_value(value, expected_value):
    if value is None or expected_value is None:
      assert value == expected_value
      return
    assert type(value) == type(expected_value), (
        f"Type mismatch: {type(value)} != {type(expected_value)}")

    if isinstance(value, (str, int, float, bool)):
      assert value == expected_value
      return
    if isinstance(value, List):
      assert len(value) == len(expected_value)
      for actual_value, exp_value in zip(value, expected_value):
        _check_value(actual_value, exp_value)
      return
    if isinstance(value, Dict):
      value_key_set = set(value.keys())
      expected_key_set = set(expected_value.keys())
      assert expected_key_set.issubset(value_key_set)
      for key in expected_key_set:
        _check_value(value[key], expected_value[key])
      return
    raise ValueError(f"Unexpected value type: {type(value)}")

  _check_value(actual_data, expected_data)


def _verify_log_messages(
    temp_dir: str,
    expected_messages: List[Tuple[str, types.LoggingType]]
) -> None:
  """Verifies that expected log messages were written."""
  if not expected_messages:
    return

  with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
    data = [json.loads(line) for line in f]
    assert len(data) == len(expected_messages)
    for actual, (expected_msg, expected_type) in zip(data, expected_messages):
      assert actual['message'] == expected_msg
      assert actual['logging_type'] == expected_type


class TestProxDashConnectionGetterSetters:
  def test_hidden_run_key_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.hidden_run_key is None
    assert connection._proxdash_connection_state.hidden_run_key is None
    connection.hidden_run_key = 'test_key'
    assert connection.hidden_run_key == 'test_key'
    assert connection._proxdash_connection_state.hidden_run_key == 'test_key'

  def test_logging_options_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.logging_options == types.LoggingOptions()
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions())
    connection.logging_options = types.LoggingOptions(stdout=True)
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions(stdout=True))

  def test_logging_options_function(self):
    dynamic_logging_options = types.LoggingOptions()
    def get_logging_options():
      return dynamic_logging_options
    connection = proxdash.ProxDashConnection(
        get_logging_options=get_logging_options,
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.logging_options == types.LoggingOptions()
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions())
    dynamic_logging_options = types.LoggingOptions(stdout=True)
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions(stdout=True))

  def test_logging_options_function_change_via_literal(self):
    dynamic_logging_options = types.LoggingOptions()
    def get_logging_options():
      return copy.deepcopy(dynamic_logging_options)
    connection = proxdash.ProxDashConnection(
        get_logging_options=get_logging_options,
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.logging_options == types.LoggingOptions()
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions())
    dynamic_logging_options = types.LoggingOptions(stdout=True)
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions(stdout=True))

  def test_proxdash_options_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.proxdash_options == types.ProxDashOptions()
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions())
    connection.proxdash_options = types.ProxDashOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions(stdout=True))

  def test_proxdash_options_function(self):
    dynamic_proxdash_options = types.ProxDashOptions()
    def get_proxdash_options():
      return dynamic_proxdash_options
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        get_proxdash_options=get_proxdash_options,
    )
    assert connection.proxdash_options == types.ProxDashOptions()
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions())
    dynamic_proxdash_options = types.ProxDashOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions(stdout=True))

  def test_proxdash_options_function_change_via_literal(self):
    dynamic_proxdash_options = types.ProxDashOptions()
    def get_proxdash_options():
      return copy.deepcopy(dynamic_proxdash_options)
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        get_proxdash_options=get_proxdash_options,
    )
    assert connection.proxdash_options == types.ProxDashOptions()
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions())
    dynamic_proxdash_options = types.ProxDashOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions(stdout=True))

  def test_api_key_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.api_key is None
    assert connection._proxdash_connection_state.api_key is None
    connection.api_key = 'test_api_key'
    assert connection.api_key == 'test_api_key'
    assert connection._proxdash_connection_state.api_key == 'test_api_key'

  def test_api_key_env_var(self, monkeypatch):
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_env_api_key')
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.api_key == 'test_env_api_key'
    assert connection._proxdash_connection_state.api_key == 'test_env_api_key'

  def test_api_key_env_var_after_setting_none(self, monkeypatch):
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_env_api_key')
    connection = proxdash.ProxDashConnection(
        api_key='test_api_key',
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.api_key == 'test_api_key'
    assert connection._proxdash_connection_state.api_key == 'test_api_key'
    connection.api_key = None
    assert connection.api_key == 'test_env_api_key'
    assert connection._proxdash_connection_state.api_key == 'test_env_api_key'

  def test_key_info_from_proxdash_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.key_info_from_proxdash is None
    assert connection._proxdash_connection_state.key_info_from_proxdash is None
    connection.key_info_from_proxdash = {'permission': 'ALL'}
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert (
        connection._proxdash_connection_state.key_info_from_proxdash ==
        {'permission': 'ALL'})

  def test_experiment_path_literal(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    connection.experiment_path = 'test/path'
    assert connection.experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'

  def test_experiment_path_literal_with_invalid_path(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    with pytest.raises(ValueError):
      connection.experiment_path = '!@#$invalid/path'

  def test_experiment_path_function(self):
    dynamic_experiment_path = '(not set)'
    def get_experiment_path():
      return dynamic_experiment_path
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
        get_experiment_path=get_experiment_path,
    )
    assert connection.experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    dynamic_experiment_path = 'test/path'
    assert connection.experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    dynamic_experiment_path = None
    assert connection.experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'

  def test_experiment_path_function_override(self):
    dynamic_experiment_path = 'test/dynamic/path'
    def get_experiment_path():
      return dynamic_experiment_path
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
        get_experiment_path=get_experiment_path,
    )
    assert connection.experiment_path == 'test/dynamic/path'
    assert (
        connection._proxdash_connection_state.experiment_path ==
        'test/dynamic/path')
    connection.experiment_path = 'test/literal/path'
    assert connection.experiment_path == 'test/literal/path'
    assert (
        connection._proxdash_connection_state.experiment_path ==
        'test/literal/path')
    connection.experiment_path = None
    assert connection.experiment_path == 'test/dynamic/path'
    assert (
        connection._proxdash_connection_state.experiment_path ==
        'test/dynamic/path')
    connection.experiment_path = '(not set)'
    assert connection.experiment_path == 'test/dynamic/path'
    assert (
        connection._proxdash_connection_state.experiment_path ==
        'test/dynamic/path')

  def test_connected_experiment_path_literal(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_connected_experiment_path_literal_logging_path')
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
    )
    connection.status = types.ProxDashConnectionStatus.CONNECTED
    assert connection.connected_experiment_path == None
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        None)

    connection.connected_experiment_path = 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path')

    connection.connected_experiment_path = None
    assert connection.connected_experiment_path == None
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        None)

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 4
      assert data[0]['logging_type'] == 'ERROR'
      assert data[0]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.')
      assert data[1]['logging_type'] == 'INFO'
      assert data[1]['message'] == 'Connected to ProxDash.'
      assert data[2]['logging_type'] == 'INFO'
      assert data[2]['message'] == 'Connected to ProxDash experiment: test/path'
      assert data[3]['logging_type'] == 'INFO'
      assert data[3]['message'] == 'Connected to ProxDash experiment: (not set)'

  def test_connected_experiment_path_literal_with_not_connected_status(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    with pytest.raises(ValueError):
      connection.connected_experiment_path = 'test/path'

  def test_status_literal(self):
    temp_dir, temp_dir_obj = _get_path_dir('test_status_literal_logging_path')
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND)
    connection.status=types.ProxDashConnectionStatus.INITIALIZING
    assert connection.status == types.ProxDashConnectionStatus.INITIALIZING
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.INITIALIZING)
    connection.status = types.ProxDashConnectionStatus.DISABLED
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.DISABLED)
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND)
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_VALID)
    connection.status = types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN)
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN)
    connection.status = types.ProxDashConnectionStatus.CONNECTED
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.CONNECTED)

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 7
      assert data[0]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.')
      assert data[1]['message'] == 'ProxDash connection initializing.'
      assert data[2]['message'] == 'ProxDash connection disabled.'
      assert data[3]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.')
      assert data[4]['message'] == (
          'ProxDash API key not valid. Please provide a valid API key.\n'
          'Check proxai.co/dashboard/api-keys page to get your API '
          'key.')
      assert data[5]['message'] == (
          'ProxDash returned an invalid response.\nPlease report this '
          'issue to the https://github.com/proxai/proxai.\n'
          'Also, please check latest stable version of ProxAI.')
      assert data[6]['message'] == 'Connected to ProxDash.'


class TestProxDashConnectionInit:
  def test_init_literals(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_init_literals_logging_path')
    connection = proxdash.ProxDashConnection(
        hidden_run_key='test_hidden_run_key',
        api_key='test_api_key',
        experiment_path='test/path',
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(stdout=True),
    )
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key == 'test_api_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(
        logging_path=temp_dir)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key='test_api_key',
            experiment_path='test/path',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(stdout=True),
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 2
      assert data[0]['message'] == 'Connected to ProxDash.'
      assert data[1]['message'] == 'Connected to ProxDash experiment: test/path'

  def test_init_literals_with_disabled_proxdash(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_init_literals_logging_path')
    connection = proxdash.ProxDashConnection(
        hidden_run_key='test_hidden_run_key',
        api_key='test_api_key',
        experiment_path='test/path',
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(
            stdout=True,
            disable_proxdash=True),
    )
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key == 'test_api_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(
        logging_path=temp_dir)
    assert connection.proxdash_options == types.ProxDashOptions(
        stdout=True,
        disable_proxdash=True)
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key='test_api_key',
            experiment_path='test/path',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(
                stdout=True,
                disable_proxdash=True),
            status=types.ProxDashConnectionStatus.DISABLED,
            key_info_from_proxdash=None,
            connected_experiment_path=None))

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 1
      assert data[0]['message'] == 'ProxDash connection disabled.'

  def test_init_functions(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_init_functions_logging_path')
    dynamic_logging_options = types.LoggingOptions(logging_path=temp_dir)
    dynamic_proxdash_options = types.ProxDashOptions(stdout=True)
    dynamic_experiment_path = 'test/path'
    def get_logging_options():
      return dynamic_logging_options
    def get_proxdash_options():
      return dynamic_proxdash_options
    def get_experiment_path():
      return dynamic_experiment_path
    connection = proxdash.ProxDashConnection(
        hidden_run_key='test_hidden_run_key',
        api_key='test_api_key',
        get_logging_options=get_logging_options,
        get_proxdash_options=get_proxdash_options,
        get_experiment_path=get_experiment_path,
    )
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key == 'test_api_key'
    assert connection.logging_options == dynamic_logging_options
    assert connection.proxdash_options == dynamic_proxdash_options
    assert connection.experiment_path == dynamic_experiment_path
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == dynamic_experiment_path
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key='test_api_key',
            experiment_path=dynamic_experiment_path,
            logging_options=dynamic_logging_options,
            proxdash_options=dynamic_proxdash_options,
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path=dynamic_experiment_path))

    dynamic_logging_options = types.LoggingOptions(
        logging_path=temp_dir,
        stdout=True)
    dynamic_proxdash_options = types.ProxDashOptions(stdout=False)
    dynamic_experiment_path = None

    connection.apply_state_changes()
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key == 'test_api_key'
    assert connection.logging_options == dynamic_logging_options
    assert connection.proxdash_options == dynamic_proxdash_options
    assert connection.experiment_path == '(not set)'
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == '(not set)'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key='test_api_key',
            experiment_path='(not set)',
            logging_options=dynamic_logging_options,
            proxdash_options=dynamic_proxdash_options,
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='(not set)'))

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 3
      assert data[0]['message'] == 'Connected to ProxDash.'
      assert data[1]['message'] == 'Connected to ProxDash experiment: test/path'
      assert data[2]['message'] == 'Connected to ProxDash experiment: (not set)'

  def test_init_invalid_combinations(self):
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          hidden_run_key='test_hidden_run_key',
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          api_key='test_api_key',
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          experiment_path='test/path',
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          logging_options=types.LoggingOptions(),
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          proxdash_options=types.ProxDashOptions(),
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          get_logging_options=lambda: types.LoggingOptions(),
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          get_proxdash_options=lambda: types.ProxDashOptions(),
          init_state=types.ProxDashConnectionState())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          get_experiment_path=lambda: 'test/path',
          init_state=types.ProxDashConnectionState())

    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          logging_options=types.LoggingOptions(),
          get_logging_options=lambda: types.LoggingOptions())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          proxdash_options=types.ProxDashOptions(),
          get_proxdash_options=lambda: types.ProxDashOptions())
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          experiment_path='test/path',
          get_experiment_path=lambda: 'test/path')

  def test_init_state_with_none_values(self):
    init_state = types.ProxDashConnectionState()
    connection = proxdash.ProxDashConnection(init_state=init_state)
    assert connection.hidden_run_key is None
    assert connection.api_key is None
    assert connection.experiment_path == '(not set)'
    assert connection.logging_options is None
    assert connection.proxdash_options is None
    assert connection.status == types.ProxDashConnectionStatus.INITIALIZING
    assert connection.key_info_from_proxdash is None
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key=None,
            api_key=None,
            experiment_path='(not set)',
            logging_options=None,
            proxdash_options=None,
            status=types.ProxDashConnectionStatus.INITIALIZING,
            key_info_from_proxdash=None,
            connected_experiment_path=None))

  def test_init_state_with_values(self):
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        api_key='test_api_key',
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(stdout=True),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_state=init_state)
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key == 'test_api_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key='test_api_key',
            experiment_path='test/path',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(stdout=True),
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

  def test_init_state_with_broken_values(self):
    # Note: This test ensures that init_state is used as is,
    # without any modifications.
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        api_key=None,
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(stdout=True),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_state=init_state)
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.api_key is None
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key',
            api_key=None,
            experiment_path='test/path',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(stdout=True),
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))


class TestProxDashConnectionGetState:
  def test_get_state(self):
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        api_key='test_api_key',
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(stdout=True),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_state=init_state)
    assert connection.get_state() == init_state


class TestProxDashConnectionUpdateState:
  def test_hidden_run_key(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.hidden_run_key is None
    assert (
        connection._proxdash_connection_state.hidden_run_key is None)
    connection.apply_state_changes(
        types.ProxDashConnectionState(hidden_run_key='test_key'))
    assert connection.hidden_run_key == 'test_key'
    assert (
        connection._proxdash_connection_state.hidden_run_key ==
        'test_key')

  def test_logging_options(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.logging_options == types.LoggingOptions()
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions())
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            logging_options=types.LoggingOptions(stdout=True)))
    assert connection.logging_options.stdout == True
    assert (
        connection._proxdash_connection_state.logging_options.stdout == True)

  def test_logging_options_function(self):
    dynamic_logging_options = types.LoggingOptions(stdout=True)
    def get_logging_options():
      return dynamic_logging_options
    connection = proxdash.ProxDashConnection(
        get_logging_options=get_logging_options,
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.logging_options == dynamic_logging_options
    assert (
        connection._proxdash_connection_state.logging_options ==
        dynamic_logging_options)

    connection.apply_state_changes(
        types.ProxDashConnectionState(
            logging_options=types.LoggingOptions(stdout=False)))
    assert connection.logging_options.stdout == False
    assert (
        connection._proxdash_connection_state.logging_options.stdout == False)

    # Note: Current implementation does not allow to set logging_options to
    # None. This is a workaround to ensure that the logging_options is None.
    connection.logging_options = None
    assert connection.logging_options == dynamic_logging_options
    assert (
        connection._proxdash_connection_state.logging_options ==
        dynamic_logging_options)

  def test_proxdash_options(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.proxdash_options == types.ProxDashOptions()
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions())
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            proxdash_options=types.ProxDashOptions(stdout=True)))
    assert connection.proxdash_options.stdout == True
    assert (
        connection._proxdash_connection_state.proxdash_options.stdout == True)

  def test_proxdash_options_with_disabled_proxdash(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_proxdash_options_with_disabled_proxdash_logging_path')
    connection = proxdash.ProxDashConnection(
        api_key='test_api_key',
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.api_key == 'test_api_key'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            api_key='test_api_key',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(),
            status=types.ProxDashConnectionStatus.CONNECTED,
            experiment_path='(not set)',
            connected_experiment_path='(not set)',
            key_info_from_proxdash={'permission': 'ALL'}))

    connection.apply_state_changes(
        types.ProxDashConnectionState(
            proxdash_options=types.ProxDashOptions(disable_proxdash=True)))
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.api_key == 'test_api_key'
    assert connection.proxdash_options.disable_proxdash == True
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            api_key='test_api_key',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(disable_proxdash=True),
            status=types.ProxDashConnectionStatus.DISABLED,
            experiment_path='(not set)'))

    connection.apply_state_changes(
        types.ProxDashConnectionState(
            proxdash_options=types.ProxDashOptions(disable_proxdash=False)))
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.api_key == 'test_api_key'
    assert connection.proxdash_options.disable_proxdash == False
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            api_key='test_api_key',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(disable_proxdash=False),
            status=types.ProxDashConnectionStatus.CONNECTED,
            experiment_path='(not set)',
            connected_experiment_path='(not set)',
            key_info_from_proxdash={'permission': 'ALL'}))

  def test_proxdash_options_function(self):
    dynamic_proxdash_options = types.ProxDashOptions(stdout=True)
    def get_proxdash_options():
      return dynamic_proxdash_options
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        get_proxdash_options=get_proxdash_options,
    )
    assert connection.proxdash_options == dynamic_proxdash_options
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        dynamic_proxdash_options)
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            proxdash_options=types.ProxDashOptions(stdout=False)))
    assert connection.proxdash_options.stdout == False
    assert (
        connection._proxdash_connection_state.proxdash_options.stdout == False)

    # Note: Current implementation does not allow to set proxdash_options to
    # None. This is a workaround to ensure that the proxdash_options is None.
    connection.proxdash_options = None
    assert connection.proxdash_options == dynamic_proxdash_options
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        dynamic_proxdash_options)

  def test_api_key(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.api_key is None
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND)
    assert connection.key_info_from_proxdash is None
    assert (
        connection._proxdash_connection_state.key_info_from_proxdash is None)
    connection.apply_state_changes(
        types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.api_key == 'test_api_key'
    assert (
        connection._proxdash_connection_state.api_key ==
        'test_api_key')
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert (
        connection._proxdash_connection_state.key_info_from_proxdash ==
        {'permission': 'ALL'})

  def test_api_key_with_disabled_proxdash(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(disable_proxdash=True),
    )
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.api_key is None
    assert connection.key_info_from_proxdash is None
    connection.apply_state_changes(
        types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.api_key == 'test_api_key'
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None

  def test_api_key_with_disabled_proxdash_with_env_api_key(self, monkeypatch):
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_env_api_key')
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(disable_proxdash=True)
    )
    assert connection.api_key == 'test_env_api_key'
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None

    connection.apply_state_changes(
        types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.api_key == 'test_api_key'
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None

    # Note: Current implementation does not allow to set api_key to None.
    # This is a workaround to ensure that the api_key is None.
    connection.api_key = None
    assert connection.api_key == 'test_env_api_key'
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None

  def test_api_key_not_found(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.api_key is None
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.key_info_from_proxdash is None

  def test_api_key_not_valid(self, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='false',
        status_code=201,
    )
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    with pytest.raises(ValueError):
      connection.apply_state_changes(
          types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.api_key == 'test_api_key'
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    assert connection.key_info_from_proxdash is None

  def test_api_key_invalid_response(self, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='Some invalid response',
        status_code=201,
    )
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    with pytest.raises(ValueError):
      connection.apply_state_changes(
          types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.api_key == 'test_api_key'
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN)
    assert connection.key_info_from_proxdash is None

  def test_key_info_from_proxdash(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.key_info_from_proxdash is None
    connection.apply_state_changes(
        types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}

    connection.apply_state_changes(
        types.ProxDashConnectionState(
            key_info_from_proxdash={'permission': 'READ'}))
    assert connection.key_info_from_proxdash == {'permission': 'READ'}

  def test_experiment_path(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_experiment_path_logging_path')
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
        experiment_path='test/path',
    )
    # 1 - Initial state with not connected experiment path:
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path is None
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path is None)
    # 2 - Set API key and check connected experiment path:
    connection.apply_state_changes(
        types.ProxDashConnectionState(api_key='test_api_key'))
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path')
    # 3 - Set same value to check info logging is not repeated:
    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path='test/path'))
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path')
    # 4 - Set different value to check info logging is repeated:
    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path='test/path2'))
    assert connection.experiment_path == 'test/path2'
    assert connection.connected_experiment_path == 'test/path2'
    assert connection._proxdash_connection_state.experiment_path == 'test/path2'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path2')
    # 5 - Set to same different value to check info logging is not repeated:
    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path='test/path2'))
    assert connection.experiment_path == 'test/path2'
    assert connection.connected_experiment_path == 'test/path2'
    assert connection._proxdash_connection_state.experiment_path == 'test/path2'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path2')
    # 6 - Set to None to check info logging is called:
    # Note: This needs to be done via setter because update_state does not
    # accept None as a value in current implementation.
    connection.experiment_path = None
    connection.apply_state_changes(types.ProxDashConnectionState())
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        '(not set)')
    # 7 - Set to same None value to check info logging is not repeated:
    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path=None))
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        '(not set)')
    # 8 - Set to initial test/path value to check info logging is called:
    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path='test/path'))
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path')

    with open(os.path.join(temp_dir, 'merged.log'), 'r') as f:
      data = [json.loads(line) for line in f
              if 'Connected to ProxDash experiment:' in line]
      assert len(data) == 4
      assert (
          data[0]['message'] == 'Connected to ProxDash experiment: test/path')
      assert (
          data[1]['message'] == 'Connected to ProxDash experiment: test/path2')
      assert (
          data[2]['message'] == 'Connected to ProxDash experiment: (not set)')
      assert (
          data[3]['message'] == 'Connected to ProxDash experiment: test/path')

  def test_experiment_path_with_api_key_not_found(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path is None
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    assert (
        connection._proxdash_connection_state.connected_experiment_path is None)

    connection.apply_state_changes(
        types.ProxDashConnectionState(experiment_path='test/path'))
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path is None
    assert connection._proxdash_connection_state.experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path is None)

  def test_connected_experiment_path(self):
    # Note: Updating connected_experiment_path is useful for most of the cases.
    # This is because connected_experiment_path is derived from experiment_path
    # if ProxDashConnection is connected.
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(),
            status=types.ProxDashConnectionStatus.API_KEY_NOT_FOUND,
            experiment_path='(not set)',
            connected_experiment_path=None,
            key_info_from_proxdash=None))

    # Note: connected_experiment_path is not changed because it is not
    # connected.
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            connected_experiment_path='test/connected/path'))
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(),
            status=types.ProxDashConnectionStatus.API_KEY_NOT_FOUND,
            experiment_path='(not set)',
            connected_experiment_path=None,
            key_info_from_proxdash=None))

    # Note: connected_experiment_path is not changed because experiment_path
    # overrides it.
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            api_key='test_api_key',
            connected_experiment_path='test/connected/path'))
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.experiment_path == '(not set)'
    assert connection.connected_experiment_path == '(not set)'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            api_key='test_api_key',
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(),
            status=types.ProxDashConnectionStatus.CONNECTED,
            experiment_path='(not set)',
            connected_experiment_path='(not set)',
            key_info_from_proxdash={'permission': 'ALL'}))

    # Note: connected_experiment_path is not changed because experiment_path
    # overrides it.
    connection.apply_state_changes(
        types.ProxDashConnectionState(
            experiment_path='test/path',
            connected_experiment_path='test/connected/path'))
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.experiment_path == 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state ==
        types.ProxDashConnectionState(
            api_key='test_api_key',
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(),
            status=types.ProxDashConnectionStatus.CONNECTED,
            experiment_path='test/path',
            connected_experiment_path='test/path',
            key_info_from_proxdash={'permission': 'ALL'}))


class TestProxDashConnectionHideSensitiveContent:
  def test_hide_sensitive_content_logging_record(self):
    connection = proxdash.ProxDashConnection(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )

    # Create a sample logging record with all fields populated
    query_record = types.QueryRecord(
        prompt="sensitive prompt",
        system="sensitive system message",
        messages=[
            {"role": "user", "content": "sensitive user message"},
            {"role": "assistant", "content": "sensitive assistant message"}
        ],
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        call_type="completion",
        max_tokens=100,
        temperature=0.7,
        stop=None,
        hash_value="test_hash"
    )

    response_record = types.QueryResponseRecord(
        response="sensitive response",
        error=None,
        error_traceback=None,
        start_utc_date=datetime.datetime.utcnow(),
        end_utc_date=datetime.datetime.utcnow(),
        local_time_offset_minute=0,
        response_time=datetime.timedelta(seconds=1),
        estimated_cost=0.0
    )

    logging_record = types.LoggingRecord(
        query_record=query_record,
        response_record=response_record,
        response_source=types.ResponseSource.PROVIDER,
        look_fail_reason=None
    )

    # Hide sensitive content
    hidden_record = connection._hide_sensitive_content_logging_record(
        logging_record)

    # Original record should not be modified
    assert logging_record.query_record.prompt == "sensitive prompt"
    assert logging_record.query_record.system == "sensitive system message"
    assert logging_record.query_record.messages == [
        {"role": "user", "content": "sensitive user message"},
        {"role": "assistant", "content": "sensitive assistant message"}
    ]
    assert logging_record.response_record.response == "sensitive response"

    # Hidden record should have sensitive content replaced
    assert hidden_record.query_record.prompt == "<sensitive content hidden>"
    assert hidden_record.query_record.system == "<sensitive content hidden>"
    assert hidden_record.query_record.messages == [{
        "role": "assistant",
        "content": "<sensitive content hidden>"
    }]
    assert (
        hidden_record.response_record.response ==
        "<sensitive content hidden>")

    # Non-sensitive fields should remain unchanged
    assert (
        hidden_record.query_record.provider_model ==
        logging_record.query_record.provider_model)
    assert (
        hidden_record.query_record.call_type ==
        logging_record.query_record.call_type)
    assert (
        hidden_record.query_record.max_tokens ==
        logging_record.query_record.max_tokens)
    assert (
        hidden_record.query_record.temperature ==
        logging_record.query_record.temperature)
    assert (
        hidden_record.query_record.stop ==
        logging_record.query_record.stop)
    assert (
        hidden_record.query_record.hash_value ==
        logging_record.query_record.hash_value)
    assert (
        hidden_record.response_record.error ==
        logging_record.response_record.error)
    assert (
        hidden_record.response_record.error_traceback ==
        logging_record.response_record.error_traceback)
    assert (
        hidden_record.response_record.start_utc_date ==
        logging_record.response_record.start_utc_date)
    assert (
        hidden_record.response_record.end_utc_date ==
        logging_record.response_record.end_utc_date)
    assert (
        hidden_record.response_record.local_time_offset_minute ==
        logging_record.response_record.local_time_offset_minute)
    assert (
        hidden_record.response_record.response_time ==
        logging_record.response_record.response_time)
    assert (
        hidden_record.response_record.estimated_cost ==
        logging_record.response_record.estimated_cost)
    assert (
        hidden_record.response_source ==
        logging_record.response_source)
    assert (
        hidden_record.look_fail_reason ==
        logging_record.look_fail_reason)


class TestProxDashConnectionUploadLoggingRecord:
  def test_upload_when_not_connected(self, requests_mock):
    """Tests that nothing happens when connection status is not CONNECTED."""
    connection, temp_dir, temp_dir_obj = _create_connection(
        status=types.ProxDashConnectionStatus.API_KEY_NOT_FOUND)
    logging_record = _create_test_logging_record()
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(requests_mock)
    _verify_log_messages(temp_dir, [])

  def test_upload_with_hide_sensitive_content(self, requests_mock):
    """Tests that sensitive content is properly hidden when
    proxdash_options.hide_sensitive_content is True."""
    connection, temp_dir, temp_dir_obj = _create_connection(
        hide_sensitive_content=True)
    logging_record = _create_test_logging_record(
        prompt="sensitive prompt",
        system="sensitive system",
        messages=[
            {"role": "user", "content": "sensitive user message"},
            {"role": "assistant", "content": "sensitive assistant message"}
        ],
        response="sensitive response"
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock, {
            'prompt': '<sensitive content hidden>',
            'system': '<sensitive content hidden>',
            'messages': [{"role": "assistant",
                          "content": "<sensitive content hidden>"}],
            'response': '<sensitive content hidden>'})
    _verify_log_messages(temp_dir, [])

  def test_upload_without_hide_sensitive_content(self, requests_mock):
    """Tests that sensitive content is preserved when hide_sensitive_content is False."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    logging_record = _create_test_logging_record(
        prompt="test prompt",
        system="test system",
        messages=[
            {"role": "user", "content": "test user message"},
            {"role": "assistant", "content": "test assistant message"}
        ],
        response="test response"
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock, {
            'prompt': 'test prompt',
            'system': 'test system',
            'messages': [
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}],
            'response': 'test response'})
    _verify_log_messages(temp_dir, [])

  def test_upload_with_stop_parameter_conversion(self, requests_mock):
    """Tests that stop parameter is properly converted to string."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    test_cases = [
        ("stop string", "stop string"),  # String
        (["stop1", "stop2"], ["stop1", "stop2"]),  # List of strings
        (None, None),  # None value
    ]
    for idx, (stop_input, expected_stop) in enumerate(test_cases):
        logging_record = _create_test_logging_record(stop=stop_input)
        requests_mock.post(
            'https://proxainest-production.up.railway.app/logging-record',
            text='success',
            status_code=201
        )
        connection.upload_logging_record(logging_record)
        _verify_proxdash_request(
            requests_mock,
            {'stop': expected_stop},
            request_id=idx)
    _verify_log_messages(temp_dir, [])

  def test_upload_with_all_permission(self, requests_mock):
    """Tests that all fields including sensitive ones are uploaded when permission is 'ALL'."""
    connection, temp_dir, temp_dir_obj = _create_connection(permission="ALL")
    logging_record = _create_test_logging_record(
        prompt="test prompt",
        system="test system",
        messages=[
            {"role": "user", "content": "test user message"},
            {"role": "assistant", "content": "test assistant message"}
        ],
        response="test response"
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock,
        {
            'prompt': 'test prompt',
            'system': 'test system',
            'messages': [
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}],
            'response': 'test response',
            'provider': 'mock_provider',
            'model': 'mock_model',
            'callType': 'GENERATE_TEXT',
            'maxTokens': 100,
            'temperature': 0.7,
            'hashValue': 'test_hash'
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_limited_permission(self, requests_mock):
    """Tests that sensitive fields are excluded when permission is not 'ALL'."""
    connection, temp_dir, temp_dir_obj = _create_connection(permission="NO_PROMPT")
    logging_record = _create_test_logging_record(
        prompt="sensitive prompt",
        system="sensitive system",
        messages=[
            {"role": "user", "content": "sensitive user message"},
            {"role": "assistant", "content": "sensitive assistant message"}
        ],
        response="sensitive response"
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock, {
            'prompt': '<sensitive content hidden>',
            'system': '<sensitive content hidden>',
            'messages': [{"role": "assistant",
                          "content": "<sensitive content hidden>"}],
            'response': '<sensitive content hidden>'})
    _verify_log_messages(temp_dir, [])

  def test_upload_failed_response(self, requests_mock):
    """Tests error logging when upload fails (non-201 status or non-success response)."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    logging_record = _create_test_logging_record()
    # Test cases for failed responses
    test_cases = [
        (400, "Bad Request", "ProxDash could not log the record. Error message:\nBad Request"),
        (201, "error", "ProxDash could not log the record. Error message:\nerror"),
        (500, "Internal Server Error", "ProxDash could not log the record. Error message:\nInternal Server Error"),
    ]
    for status_code, response_text, expected_error in test_cases:
      if os.path.exists(os.path.join(temp_dir, 'merged.log')):
          os.remove(os.path.join(temp_dir, 'merged.log'))
      requests_mock.post(
          'https://proxainest-production.up.railway.app/logging-record',
          text=response_text,
          status_code=status_code
      )
      connection.upload_logging_record(logging_record)
      _verify_log_messages(
          temp_dir,
          [(expected_error, types.LoggingType.ERROR)]
      )

  def test_upload_with_none_values(self, requests_mock):
    """Tests handling of None values in various fields of the logging record."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    logging_record = _create_test_logging_record(
        prompt=None,
        system=None,
        messages=None,
        response=None,
        error=None,
        error_traceback=None,
        stop=None
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock,
        {
            'prompt': None,
            'system': None,
            'messages': None,
            'response': None,
            'error': None,
            'errorTraceback': None,
            'stop': None
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_datetime_conversion(self, requests_mock):
    """Tests proper conversion of datetime objects to ISO format."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    test_start_time = datetime.datetime(
        2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    test_end_time = datetime.datetime(
        2024, 1, 1, 12, 1, tzinfo=datetime.timezone.utc)
    logging_record = _create_test_logging_record()
    logging_record.response_record.start_utc_date = test_start_time
    logging_record.response_record.end_utc_date = test_end_time
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock,
        {
            'startUTCDate': '2024-01-01T12:00:00+00:00',
            'endUTCDate': '2024-01-01T12:01:00+00:00'
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_response_time_conversion(self, requests_mock):
    """Tests conversion of timedelta to milliseconds for response_time."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    test_cases = [
        (datetime.timedelta(seconds=1), 1000.0),  # 1 second = 1000ms
        (datetime.timedelta(milliseconds=500), 500.0),  # 500ms
        (datetime.timedelta(microseconds=1500), 1.5),  # 1.5ms
        (datetime.timedelta(minutes=1), 60000.0),  # 1 minute = 60000ms
    ]
    for idx, (delta, expected_ms) in enumerate(test_cases):
      logging_record = _create_test_logging_record()
      logging_record.response_record.response_time = delta
      requests_mock.post(
          'https://proxainest-production.up.railway.app/logging-record',
          text='success',
          status_code=201
      )
      connection.upload_logging_record(logging_record)
      _verify_proxdash_request(
          requests_mock,
          {'responseTime': expected_ms},
          request_id=idx
      )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_complete_logging_record(self, requests_mock):
    """Tests upload with a fully populated logging record containing all possible fields."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    logging_record = types.LoggingRecord(
        query_record=types.QueryRecord(
            prompt="test prompt",
            system="test system",
            messages=[
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}
            ],
            provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
            call_type=types.CallType.GENERATE_TEXT,
            max_tokens=100,
            temperature=0.7,
            stop=["stop1", "stop2"],
            hash_value="test_hash"
        ),
        response_record=types.QueryResponseRecord(
            response="test response",
            error="test error",
            error_traceback="test error traceback",
            start_utc_date=datetime.datetime(
                2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
            end_utc_date=datetime.datetime(
                2024, 1, 1, 12, 1, tzinfo=datetime.timezone.utc),
            local_time_offset_minute=120,  # UTC+2
            response_time=datetime.timedelta(seconds=1),
            estimated_cost=0.001
        ),
        response_source=types.ResponseSource.PROVIDER,
        look_fail_reason=types.CacheLookFailReason.CACHE_NOT_FOUND
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201
    )
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(
        requests_mock,
        {
            'prompt': 'test prompt',
            'system': 'test system',
            'messages': [
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}
            ],
            'provider': 'mock_provider',
            'model': 'mock_model',
            'callType': 'GENERATE_TEXT',
            'maxTokens': 100,
            'temperature': 0.7,
            'stop': ['stop1', 'stop2'],
            'hashValue': 'test_hash',
            'response': 'test response',
            'error': 'test error',
            'errorTraceback': 'test error traceback',
            'startUTCDate': '2024-01-01T12:00:00+00:00',
            'endUTCDate': '2024-01-01T12:01:00+00:00',
            'localTimeOffsetMinute': 120,
            'responseTime': 1000.0,
            'estimatedCost': 0.001,
            'responseSource': 'PROVIDER',
            'lookFailReason': 'CACHE_NOT_FOUND'
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_network_error(self, requests_mock):
    """Tests handling of network errors during upload."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    logging_record = _create_test_logging_record()
    test_cases = [
        (requests.exceptions.ConnectionError, "Connection error"),
        (requests.exceptions.Timeout, "Request timed out"),
        (requests.exceptions.RequestException, "General request error"),
    ]
    for idx, (exception_class, error_message) in enumerate(test_cases):
      if os.path.exists(os.path.join(temp_dir, 'merged.log')):
          os.remove(os.path.join(temp_dir, 'merged.log'))
      requests_mock.post(
          'https://proxainest-production.up.railway.app/logging-record',
          exc=exception_class(error_message)
      )
      connection.upload_logging_record(logging_record)
      _verify_log_messages(
          temp_dir,
          [(
              f'ProxDash could not log the record. Error message:\n{error_message}',
              types.LoggingType.ERROR
          )]
      )

  def test_upload_with_invalid_api_key(self, requests_mock):
    """Tests behavior when API key is invalid or expired."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID

    logging_record = _create_test_logging_record()
    connection.upload_logging_record(logging_record)
    _verify_proxdash_request(requests_mock)
    _verify_log_messages(temp_dir, [])


class TestProxDashConnectionScenarios:
  def test_update_state_with_none_values_init_state(self):
    init_state = types.ProxDashConnectionState()
    connection = proxdash.ProxDashConnection(init_state=init_state)
    with pytest.raises(
        ValueError,
        match='ProxDash options are not set for both old and new states. '
        'This creates an invalid state change.'):
      connection.apply_state_changes()

  def test_proxdash_options_function_change_via_literal(self, requests_mock):
    logging_record = _create_test_logging_record()
    dynamic_proxdash_options = types.ProxDashOptions()
    def get_proxdash_options():
      return copy.deepcopy(dynamic_proxdash_options)

    connection = proxdash.ProxDashConnection(
        api_key='test_api_key',
        logging_options=types.LoggingOptions(),
        get_proxdash_options=get_proxdash_options,
    )
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED

    dynamic_proxdash_options.disable_proxdash = True
    connection.upload_logging_record(logging_record)
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    _verify_proxdash_request(requests_mock)

    dynamic_proxdash_options.disable_proxdash = False
    connection.upload_logging_record(logging_record)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    _verify_proxdash_request(
        requests_mock, {
            'prompt': 'test prompt',
            'system': 'test system',
            'messages': [
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}],
            'response': 'test response'})

  def test_proxdash_options_function_change_via_reference(self, requests_mock):
    logging_record = _create_test_logging_record()
    dynamic_proxdash_options = types.ProxDashOptions()
    def get_proxdash_options():
      return dynamic_proxdash_options

    connection = proxdash.ProxDashConnection(
        api_key='test_api_key',
        logging_options=types.LoggingOptions(),
        get_proxdash_options=get_proxdash_options,
    )
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED

    dynamic_proxdash_options.disable_proxdash = True
    connection.upload_logging_record(logging_record)
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    _verify_proxdash_request(requests_mock)

    dynamic_proxdash_options.disable_proxdash = False
    connection.upload_logging_record(logging_record)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    _verify_proxdash_request(
        requests_mock, {
            'prompt': 'test prompt',
            'system': 'test system',
            'messages': [
                {"role": "user", "content": "test user message"},
                {"role": "assistant", "content": "test assistant message"}],
            'response': 'test response'})

  # def test_rapid_state_updates(self):
  #   """Tests behavior when state is updated multiple times in rapid succession"""
  #   pass

  # def test_concurrent_logging_uploads(self):
  #   """Tests behavior when multiple logging records are uploaded concurrently"""
  #   pass

  # def test_experiment_path_special_characters(self):
  #   """Tests handling of experiment paths with special characters, unicode, etc."""
  #   pass

  # def test_very_large_logging_record(self):
  #   """Tests handling of extremely large logging records (e.g., huge responses)"""
  #   pass

  # def test_network_timeout_recovery(self):
  #   """Tests recovery behavior after network timeouts during API calls"""
  #   pass

  # def test_api_key_rotation_during_operation(self):
  #   """Tests behavior when API key is changed while connection is active"""
  #   pass

  # def test_memory_leak_large_history(self):
  #   """Tests memory usage with large number of logging records"""
  #   pass

  # def test_invalid_utf8_in_responses(self):
  #   """Tests handling of invalid UTF-8 characters in API responses"""
  #   pass

  # def test_state_recovery_after_crash(self):
  #   """Tests state recovery after unexpected program termination"""
  #   pass

  # def test_mixed_permission_changes(self):
  #   """Tests behavior when API key permissions change during operation"""
  #   pass

  # def test_experiment_path_race_condition(self):
  #   """Tests race conditions between experiment path updates and logging"""
  #   pass

  # def test_nested_state_updates(self):
  #   """Tests nested/recursive state update scenarios"""
  #   pass

  # def test_zero_length_content(self):
  #   """Tests handling of empty strings and zero-length content in records"""
  #   pass

  # def test_malformed_logging_options(self):
  #   """Tests handling of malformed or partially initialized logging options"""
  #   pass

  # def test_dynamic_options_exception(self):
  #   """Tests behavior when dynamic options callbacks raise exceptions"""
  #   pass

  # def test_status_transition_edge_cases(self):
  #   """Tests unusual status transitions and edge cases"""
  #   pass

  # def test_connection_stress_test(self):
  #   """Tests connection stability under high load and rapid state changes"""
  #   pass

  # def test_invalid_experiment_path_characters(self):
  #   """Tests handling of invalid characters in experiment paths"""
  #   pass

  # def test_connection_timeout_during_upload(self):
  #   """Tests behavior when connection times out during record upload"""
  #   pass

  # def test_mixed_encoding_handling(self):
  #   """Tests handling of mixed character encodings in logging records"""
  #   pass

  # def test_rapid_connection_recycling(self):
  #   """Tests rapid creation and destruction of connections"""
  #   pass

  # def test_environment_variable_race_condition(self):
  #   """Tests race conditions with environment variable changes"""
  #   pass

  # def test_invalid_state_transitions(self):
  #   """Tests invalid state transition attempts and error handling"""
  #   pass

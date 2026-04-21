import datetime
import json
import os
import tempfile

import pydantic
import pytest
import requests
import requests_mock as requests_mock_module

import proxai.chat.chat_session as chat_session
import proxai.chat.message as message
import proxai.chat.message_content as message_content
import proxai.connections.proxdash as proxdash
import proxai.connectors.model_configs as model_configs
import proxai.types as types


class SamplePydanticModel(pydantic.BaseModel):
  """Sample pydantic model for unit tests."""
  name: str
  value: int
  active: bool = True


# Sentinel value to distinguish "not passed" from "explicitly passed as None"
_UNSET = object()


@pytest.fixture(autouse=True)
def setup_test(monkeypatch, requests_mock):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  requests_mock.get(
      'https://proxainest-production.up.railway.app/ingestion/verify-key',
      text='{"success": true, "data": {"permission": "ALL"}}',
      status_code=200,
  )
  yield


@pytest.fixture
def model_configs_instance():
  """Fixture to provide a ModelConfigs instance for testing."""
  return model_configs.ModelConfigs()


def _get_path_dir(temp_path: str):
  temp_dir = tempfile.TemporaryDirectory()
  path = os.path.join(temp_dir.name, temp_path)
  os.makedirs(path, exist_ok=True)
  return path, temp_dir


def _create_test_call_record(
    prompt: str | None = 'test prompt',
    system_prompt: str | None = 'test system',
    output_text: str | None = 'test response',
    output_format: types.OutputFormat | None = None,
    output_json: dict | None = None,
    output_pydantic: pydantic.BaseModel | None = None,
    error: str | None = None,
    error_traceback: str | None = None,
    stop: str | list[str] | None = None,
    hash_value: str = 'test_hash',
    result_source: types.ResultSource = types.ResultSource.PROVIDER,
    cache_look_fail_reason: types.CacheLookFailReason | None = None,
) -> types.CallRecord:
  """Creates a test call record with default values."""
  model_configs_instance = model_configs.ModelConfigs()
  query = types.QueryRecord(
      prompt=prompt,
      system_prompt=system_prompt,
      provider_model=model_configs_instance.get_provider_model(
          ('mock_provider', 'mock_model')
      ),
      parameters=types.ParameterType(
          max_tokens=100, temperature=0.7, stop=stop
      ),
      output_format=output_format,
      hash_value=hash_value,
  )

  result = types.ResultRecord(
      status=types.ResultStatusType.SUCCESS,
      output_text=output_text,
      output_json=output_json,
      output_pydantic=output_pydantic,
      error=error,
      error_traceback=error_traceback,
      timestamp=types.TimeStampType(
          start_utc_date=datetime.datetime(2024, 1, 1, 12, 0),
          end_utc_date=datetime.datetime(2024, 1, 1, 12, 1),
          local_time_offset_minute=0,
          response_time=datetime.timedelta(seconds=1),
      ),
      usage=types.UsageType(
          input_tokens=10, output_tokens=20, total_tokens=30, estimated_cost=5
      ),
  )

  connection = types.ConnectionMetadata(
      result_source=result_source,
      cache_look_fail_reason=cache_look_fail_reason,
  )

  return types.CallRecord(query=query, result=result, connection=connection)


def _create_connection(
    status: types.ProxDashConnectionStatus = types.ProxDashConnectionStatus.
    CONNECTED, hide_sensitive_content: bool = False, permission: str = "ALL",
    temp_dir: str | None = None
) -> tuple[proxdash.ProxDashConnection, str]:
  """Creates a ProxDashConnection."""
  if temp_dir is None:
    temp_dir, temp_dir_obj = _get_path_dir('test_upload_logging_record')

  proxdash_connection_params = proxdash.ProxDashConnectionParams(
      logging_options=types.LoggingOptions(logging_path=temp_dir),
      proxdash_options=types.ProxDashOptions(
          hide_sensitive_content=hide_sensitive_content, api_key='test_api_key'
      )
  )

  connection = proxdash.ProxDashConnection(
      init_from_params=proxdash_connection_params
  )
  connection.status = status
  connection.key_info_from_proxdash = {'permission': permission}

  return connection, temp_dir, temp_dir_obj


def _verify_proxdash_request(
    requests_mock, expected_data: dict | None = None,
    request_id: int | None = None, response_status: int = 201,
    response_text: str = 'success'
) -> None:
  """Verifies that the request to ProxDash was made with expected data."""
  logging_record_requests = [
      request for request in requests_mock.request_history if request.url ==
      'https://proxainest-production.up.railway.app/ingestion/call-records'
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
    # Handle JSON string vs list/dict comparison
    if isinstance(value, str) and isinstance(expected_value, (list, dict)):
      parsed_value = json.loads(value)
      assert parsed_value == expected_value
      return

    # Allow int/float comparison if values are equal
    if isinstance(value,
                  (int, float)) and isinstance(expected_value, (int, float)):
      assert value == expected_value
      return

    assert type(value) is type(expected_value), (
        f"Type mismatch: {type(value)} != {type(expected_value)}"
    )

    if isinstance(value, (str, bool)):
      assert value == expected_value
      return
    if isinstance(value, list):
      assert len(value) == len(expected_value)
      for actual_value, exp_value in zip(value, expected_value, strict=False):
        _check_value(actual_value, exp_value)
      return
    if isinstance(value, dict):
      value_key_set = set(value.keys())
      expected_key_set = set(expected_value.keys())
      assert expected_key_set.issubset(value_key_set)
      for key in expected_key_set:
        _check_value(value[key], expected_value[key])
      return
    raise ValueError(f"Unexpected value type: {type(value)}")

  _check_value(actual_data, expected_data)


def _verify_log_messages(
    temp_dir: str, expected_messages: list[tuple[str, types.LoggingType]]
) -> None:
  """Verifies that expected log messages were written."""
  if not expected_messages:
    return

  with open(os.path.join(temp_dir, 'merged.log')) as f:
    data = [json.loads(line) for line in f]
    assert len(data) == len(expected_messages)
    for actual, (expected_msg,
                 expected_type) in zip(data, expected_messages, strict=False):
      assert actual['message'] == expected_msg
      assert actual['logging_type'] == expected_type


class TestProxDashConnectionGetterSetters:

  def test_hidden_run_key_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.hidden_run_key is None
    assert connection._proxdash_connection_state.hidden_run_key is None
    connection.hidden_run_key = 'test_key'
    assert connection.hidden_run_key == 'test_key'
    assert connection._proxdash_connection_state.hidden_run_key == 'test_key'

  def test_logging_options_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.logging_options == types.LoggingOptions()
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions()
    )
    connection.logging_options = types.LoggingOptions(stdout=True)
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.logging_options ==
        types.LoggingOptions(stdout=True)
    )

  def test_proxdash_options_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.proxdash_options == types.ProxDashOptions()
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions()
    )
    connection.proxdash_options = types.ProxDashOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert (
        connection._proxdash_connection_state.proxdash_options ==
        types.ProxDashOptions(stdout=True)
    )

  def test_api_key_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.proxdash_options.api_key is None
    assert connection._proxdash_connection_state.proxdash_options.api_key is None
    connection.proxdash_options.api_key = 'test_api_key'
    assert connection.proxdash_options.api_key == 'test_api_key'
    assert (
        connection._proxdash_connection_state.proxdash_options.api_key ==
        'test_api_key'
    )

  def test_api_key_env_var(self, monkeypatch):
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_env_api_key')
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.proxdash_options.api_key == 'test_env_api_key'
    assert (
        connection._proxdash_connection_state.proxdash_options.api_key ==
        'test_env_api_key'
    )

  def test_key_info_from_proxdash_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.key_info_from_proxdash is None
    assert connection._proxdash_connection_state.key_info_from_proxdash is None
    connection.key_info_from_proxdash = {'permission': 'ALL'}
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert (
        connection._proxdash_connection_state.key_info_from_proxdash == {
            'permission': 'ALL'
        }
    )

  def test_experiment_path_literal(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.experiment_path == '(not set)'
    assert connection._proxdash_connection_state.experiment_path == '(not set)'
    connection.experiment_path = 'test/path'
    assert connection.experiment_path == 'test/path'
    assert connection._proxdash_connection_state.experiment_path == 'test/path'

  def test_experiment_path_literal_with_invalid_path(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    with pytest.raises(ValueError):
      connection.experiment_path = '!@#$invalid/path'

  def test_connected_experiment_path_literal(self):
    temp_dir, temp_dir_obj = _get_path_dir(
        'test_connected_experiment_path_literal_logging_path'
    )
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    connection.status = types.ProxDashConnectionStatus.CONNECTED
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state.connected_experiment_path is None
    )

    connection.connected_experiment_path = 'test/path'
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state.connected_experiment_path ==
        'test/path'
    )

    connection.connected_experiment_path = None
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state.connected_experiment_path is None
    )

    with open(os.path.join(temp_dir, 'merged.log')) as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 4
      assert data[0]['logging_type'] == 'ERROR'
      assert data[0]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.'
      )
      assert data[1]['logging_type'] == 'INFO'
      assert data[1]['message'].startswith('Connected to ProxDash at ')
      assert data[2]['logging_type'] == 'INFO'
      assert data[2]['message'] == 'Connected to ProxDash experiment: test/path'
      assert data[3]['logging_type'] == 'INFO'
      assert data[3]['message'] == 'Connected to ProxDash experiment: (not set)'

  def test_connected_experiment_path_literal_with_not_connected_status(self):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    with pytest.raises(ValueError):
      connection.connected_experiment_path = 'test/path'

  def test_status_literal(self):
    temp_dir, temp_dir_obj = _get_path_dir('test_status_literal_logging_path')
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    )
    connection.status = types.ProxDashConnectionStatus.INITIALIZING
    assert connection.status == types.ProxDashConnectionStatus.INITIALIZING
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.INITIALIZING
    )
    connection.status = types.ProxDashConnectionStatus.DISABLED
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.DISABLED
    )
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    )
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    assert connection.status == types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    )
    connection.status = types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
    )
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN
    )
    connection.status = types.ProxDashConnectionStatus.CONNECTED
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert (
        connection._proxdash_connection_state.status ==
        types.ProxDashConnectionStatus.CONNECTED
    )

    with open(os.path.join(temp_dir, 'merged.log')) as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 7
      assert data[0]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.'
      )
      assert data[1]['message'] == 'ProxDash connection initializing.'
      assert data[2]['message'] == 'ProxDash connection disabled.'
      assert data[3]['message'] == (
          'ProxDash connection disabled. Please provide a valid API key '
          'either as an argument or as an environment variable.'
      )
      assert data[4]['message'] == (
          'ProxDash API key not valid. Please provide a valid API key.\n'
          'Check proxai.co/dashboard/api-keys page to get your API '
          'key.'
      )
      assert data[5]['message'] == (
          'ProxDash returned an invalid response.\nPlease report this '
          'issue to the https://github.com/proxai/proxai.\n'
          'Also, please check latest stable version of ProxAI.'
      )
      assert data[6]['message'].startswith('Connected to ProxDash at ')


class TestProxDashConnectionInit:

  def test_init_literals(self):
    temp_dir, temp_dir_obj = _get_path_dir('test_init_literals_logging_path')
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        hidden_run_key='test_hidden_run_key',
        experiment_path='test/path',
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(
            stdout=True, api_key='test_api_key'
        ),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.proxdash_options.api_key == 'test_api_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(
        logging_path=temp_dir
    )
    assert connection.proxdash_options == types.ProxDashOptions(
        stdout=True, api_key='test_api_key'
    )
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state == types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key', experiment_path='test/path',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(
                stdout=True, api_key='test_api_key'
            ), status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'
        )
    )

    with open(os.path.join(temp_dir, 'merged.log')) as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 2
      assert data[0]['message'].startswith('Connected to ProxDash at ')
      assert data[1]['message'] == 'Connected to ProxDash experiment: test/path'

  def test_init_literals_with_disabled_proxdash(self):
    temp_dir, temp_dir_obj = _get_path_dir('test_init_literals_logging_path')
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        hidden_run_key='test_hidden_run_key',
        experiment_path='test/path',
        logging_options=types.LoggingOptions(logging_path=temp_dir),
        proxdash_options=types.ProxDashOptions(
            stdout=True, disable_proxdash=True, api_key='test_api_key'
        ),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.proxdash_options.api_key == 'test_api_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(
        logging_path=temp_dir
    )
    assert connection.proxdash_options == types.ProxDashOptions(
        stdout=True, disable_proxdash=True, api_key='test_api_key'
    )
    assert connection.status == types.ProxDashConnectionStatus.DISABLED
    assert connection.key_info_from_proxdash is None
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state == types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key', experiment_path='test/path',
            logging_options=types.LoggingOptions(logging_path=temp_dir),
            proxdash_options=types.ProxDashOptions(
                stdout=True, disable_proxdash=True, api_key='test_api_key'
            ), status=types.ProxDashConnectionStatus.DISABLED,
            key_info_from_proxdash=None, connected_experiment_path=None
        )
    )

    with open(os.path.join(temp_dir, 'merged.log')) as f:
      data = [json.loads(line) for line in f]
      assert len(data) == 1
      assert data[0]['message'] == 'ProxDash connection disabled.'

  def test_init_invalid_combinations(self):
    with pytest.raises(ValueError):
      proxdash.ProxDashConnection(
          init_from_params=proxdash.ProxDashConnectionParams(),
          init_from_state=types.ProxDashConnectionState()
      )

  def test_init_state_with_none_values(self):
    init_state = types.ProxDashConnectionState()
    connection = proxdash.ProxDashConnection(init_from_state=init_state)
    assert connection.hidden_run_key is None
    assert connection.experiment_path == '(not set)'
    assert connection.logging_options == types.LoggingOptions()
    assert connection.proxdash_options == types.ProxDashOptions()
    assert connection.status is None
    assert connection.key_info_from_proxdash is None
    assert connection.connected_experiment_path is None
    assert (
        connection._proxdash_connection_state == types.ProxDashConnectionState(
            hidden_run_key=None, experiment_path='(not set)',
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(), status=None,
            key_info_from_proxdash=None, connected_experiment_path=None
        )
    )

  def test_init_state_with_values(self):
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(
            stdout=True, api_key='test_api_key'
        ),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_from_state=init_state)
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(
        stdout=True, api_key='test_api_key'
    )
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state == types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key', experiment_path='test/path',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(
                stdout=True, api_key='test_api_key'
            ), status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'
        )
    )

  def test_init_state_with_broken_values(self):
    # Note: This test ensures that init_state is used as is,
    # without any modifications.
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(stdout=True, api_key=None),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_from_state=init_state)
    assert connection.hidden_run_key == 'test_hidden_run_key'
    assert connection.experiment_path == 'test/path'
    assert connection.logging_options == types.LoggingOptions(stdout=True)
    assert connection.proxdash_options == types.ProxDashOptions(stdout=True)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED
    assert connection.key_info_from_proxdash == {'permission': 'ALL'}
    assert connection.connected_experiment_path == 'test/path'
    assert (
        connection._proxdash_connection_state == types.ProxDashConnectionState(
            hidden_run_key='test_hidden_run_key', experiment_path='test/path',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(stdout=True),
            status=types.ProxDashConnectionStatus.CONNECTED,
            key_info_from_proxdash={'permission': 'ALL'
                                   }, connected_experiment_path='test/path'
        )
    )


class TestProxDashConnectionGetState:

  def test_get_state(self):
    init_state = types.ProxDashConnectionState(
        hidden_run_key='test_hidden_run_key',
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_options=types.ProxDashOptions(
            stdout=True, api_key='test_api_key'
        ),
        status=types.ProxDashConnectionStatus.CONNECTED,
        experiment_path='test/path',
        key_info_from_proxdash={'permission': 'ALL'},
        connected_experiment_path='test/path',
    )
    connection = proxdash.ProxDashConnection(init_from_state=init_state)
    assert connection.get_state() == init_state


class TestProxDashConnectionHideSensitiveContent:

  def test_hide_sensitive_content(self, model_configs_instance):
    proxdash_connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(),
    )
    connection = proxdash.ProxDashConnection(
        init_from_params=proxdash_connection_params
    )

    call_record = _create_test_call_record(
        prompt='sensitive prompt',
        system_prompt='sensitive system',
        output_text='sensitive response',
    )

    hidden_record = connection._hide_sensitive_content(call_record)

    # Original record should not be modified
    assert call_record.query.prompt == 'sensitive prompt'
    assert call_record.query.system_prompt == 'sensitive system'
    assert call_record.result.output_text == 'sensitive response'

    # Hidden record should have sensitive content replaced
    hidden_str = proxdash._SENSITIVE_CONTENT_HIDDEN_STRING
    assert hidden_record.query.prompt == hidden_str
    assert hidden_record.query.system_prompt == hidden_str
    assert hidden_record.query.chat is None
    assert hidden_record.result.output_text == hidden_str
    assert hidden_record.result.output_image is None
    assert hidden_record.result.output_audio is None
    assert hidden_record.result.output_video is None
    assert hidden_record.result.output_json is None
    assert hidden_record.result.output_pydantic is None
    assert hidden_record.result.content is None
    assert hidden_record.result.choices is None

    # Non-sensitive fields should remain unchanged
    assert hidden_record.query.provider_model == call_record.query.provider_model
    assert hidden_record.query.parameters == call_record.query.parameters
    assert hidden_record.query.hash_value == call_record.query.hash_value
    assert hidden_record.result.error == call_record.result.error
    assert (
        hidden_record.result.error_traceback ==
        call_record.result.error_traceback
    )
    assert hidden_record.result.timestamp == call_record.result.timestamp
    assert hidden_record.result.usage == call_record.result.usage
    assert hidden_record.connection == call_record.connection


class TestProxDashConnectionUploadCallRecord:

  def test_upload_when_not_connected(self, requests_mock):
    """Tests that nothing happens when connection status is not CONNECTED."""
    connection, temp_dir, temp_dir_obj = _create_connection(
        status=types.ProxDashConnectionStatus.API_KEY_NOT_FOUND
    )
    call_record = _create_test_call_record()
    connection.upload_call_record(call_record)
    _verify_proxdash_request(requests_mock)
    _verify_log_messages(temp_dir, [])

  def test_upload_with_hide_sensitive_content(self, requests_mock):
    """Tests that sensitive content is hidden when configured."""
    connection, temp_dir, temp_dir_obj = _create_connection(
        hide_sensitive_content=True
    )
    call_record = _create_test_call_record(
        prompt='sensitive prompt', system_prompt='sensitive system',
        output_text='sensitive response'
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    hidden_str = proxdash._SENSITIVE_CONTENT_HIDDEN_STRING
    _verify_proxdash_request(
        requests_mock, {
            'query': {
                'prompt': hidden_str,
                'system_prompt': hidden_str,
            },
            'result': {
                'output_text': hidden_str,
            },
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_without_hide_sensitive_content(self, requests_mock):
    """Tests that content is preserved when hide_sensitive_content is False."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record(
        prompt='test prompt', system_prompt='test system',
        output_text='test response'
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'query': {
                'prompt': 'test prompt',
                'system_prompt': 'test system',
            },
            'result': {
                'output_text': 'test response',
            },
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_stop_parameter(self, requests_mock):
    """Tests that stop parameter is properly serialized."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    test_cases = [
        ('stop string', 'stop string'),
        (['stop1', 'stop2'], ['stop1', 'stop2']),
    ]
    for idx, (stop_input, expected_stop) in enumerate(test_cases):
      call_record = _create_test_call_record(stop=stop_input)
      requests_mock.post(
          'https://proxainest-production.up.railway.app/ingestion/call-records',
          text='{"success": true}', status_code=201
      )
      connection.upload_call_record(call_record)
      _verify_proxdash_request(
          requests_mock, {'query': {
              'parameters': {
                  'stop': expected_stop
              }
          }}, request_id=idx
      )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_all_permission(self, requests_mock):
    """Tests that all fields are uploaded when permission is 'ALL'."""
    connection, temp_dir, temp_dir_obj = _create_connection(permission='ALL')
    call_record = _create_test_call_record(
        prompt='test prompt', system_prompt='test system',
        output_text='test response'
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'schema_version': 2,
            'query': {
                'prompt': 'test prompt',
                'system_prompt': 'test system',
                'provider_model': {
                    'provider': 'mock_provider',
                    'model': 'mock_model',
                },
                'parameters': {
                    'max_tokens': 100,
                    'temperature': 0.7,
                },
                'hash_value': 'test_hash',
            },
            'result': {
                'output_text': 'test response',
            },
            'connection': {
                'result_source': 'PROVIDER',
                'caller_type': 'SDK',
                'caller_app': 'PYTHON_SDK',
            },
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_limited_permission(self, requests_mock):
    """Tests that sensitive fields are hidden when permission is 'NO_PROMPT'."""
    connection, temp_dir, temp_dir_obj = _create_connection(
        permission='NO_PROMPT'
    )
    call_record = _create_test_call_record(
        prompt='sensitive prompt', system_prompt='sensitive system',
        output_text='sensitive response'
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    hidden_str = proxdash._SENSITIVE_CONTENT_HIDDEN_STRING
    _verify_proxdash_request(
        requests_mock, {
            'query': {
                'prompt': hidden_str,
                'system_prompt': hidden_str,
            },
            'result': {
                'output_text': hidden_str,
            },
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_failed_response(self, requests_mock):
    """Tests error logging when upload fails."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record()
    test_cases = [
        (
            400, 'Bad Request',
            'ProxDash could not log the record. Status code: 400, '
            'Response: Bad Request'
        ),
        (
            201, 'error',
            'ProxDash could not log the record. Invalid JSON response: error'
        ),
        (
            500, 'Internal Server Error',
            'ProxDash could not log the record. Status code: 500, '
            'Response: Internal Server Error'
        ),
    ]
    for status_code, response_text, expected_error in test_cases:
      if os.path.exists(os.path.join(temp_dir, 'merged.log')):
        os.remove(os.path.join(temp_dir, 'merged.log'))
      requests_mock.post(
          'https://proxainest-production.up.railway.app/ingestion/call-records',
          text=response_text, status_code=status_code
      )
      connection.upload_call_record(call_record)
      _verify_log_messages(
          temp_dir, [(expected_error, types.LoggingType.ERROR)]
      )

  def test_upload_with_none_values(self, requests_mock):
    """Tests handling of None values in various fields."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record(
        prompt=None, system_prompt=None, output_text=None, error=None,
        error_traceback=None, stop=None
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    actual_data = json.loads([
        r for r in requests_mock.request_history if 'call-records' in r.url
    ][0].text)
    assert 'prompt' not in actual_data.get('query', {})
    assert 'system_prompt' not in actual_data.get('query', {})
    assert 'output_text' not in actual_data.get('result', {})
    _verify_log_messages(temp_dir, [])

  def test_upload_with_datetime_conversion(self, requests_mock):
    """Tests proper conversion of datetime objects to ISO format."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record()
    call_record.result.timestamp.start_utc_date = datetime.datetime(
        2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc
    )
    call_record.result.timestamp.end_utc_date = datetime.datetime(
        2024, 1, 1, 12, 1, tzinfo=datetime.timezone.utc
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'result': {
                'timestamp': {
                    'start_utc_date': '2024-01-01T12:00:00+00:00',
                    'end_utc_date': '2024-01-01T12:01:00+00:00',
                }
            }
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_response_time_conversion(self, requests_mock):
    """Tests conversion of timedelta to milliseconds for response_time."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    test_cases = [
        (datetime.timedelta(seconds=1), 1000),
        (datetime.timedelta(milliseconds=500), 500),
        (datetime.timedelta(microseconds=1500), 1),
        (datetime.timedelta(minutes=1), 60000),
    ]
    for idx, (delta, expected_ms) in enumerate(test_cases):
      call_record = _create_test_call_record()
      call_record.result.timestamp.response_time = delta
      requests_mock.post(
          'https://proxainest-production.up.railway.app/ingestion/call-records',
          text='{"success": true}', status_code=201
      )
      connection.upload_call_record(call_record)
      _verify_proxdash_request(
          requests_mock,
          {'result': {
              'timestamp': {
                  'response_time': expected_ms
              }
          }}, request_id=idx
      )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_complete_call_record(
      self, requests_mock, model_configs_instance
  ):
    """Tests upload with a fully populated call record."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record(
        prompt='test prompt',
        system_prompt='test system',
        output_text='test response',
        error='test error',
        error_traceback='test error traceback',
        stop=['stop1', 'stop2'],
        hash_value='test_hash',
        result_source=types.ResultSource.PROVIDER,
        cache_look_fail_reason=types.CacheLookFailReason.CACHE_NOT_FOUND,
    )
    call_record.result.timestamp.start_utc_date = datetime.datetime(
        2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc
    )
    call_record.result.timestamp.end_utc_date = datetime.datetime(
        2024, 1, 1, 12, 1, tzinfo=datetime.timezone.utc
    )
    call_record.result.timestamp.local_time_offset_minute = 120
    call_record.result.timestamp.response_time = (datetime.timedelta(seconds=1))
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'schema_version': 2,
            'query': {
                'prompt': 'test prompt',
                'system_prompt': 'test system',
                'provider_model': {
                    'provider': 'mock_provider',
                    'model': 'mock_model',
                },
                'parameters': {
                    'max_tokens': 100,
                    'temperature': 0.7,
                    'stop': ['stop1', 'stop2'],
                },
                'hash_value': 'test_hash',
            },
            'result': {
                'status': 'SUCCESS',
                'output_text': 'test response',
                'error': 'test error',
                'error_traceback': 'test error traceback',
                'timestamp': {
                    'start_utc_date': '2024-01-01T12:00:00+00:00',
                    'end_utc_date': '2024-01-01T12:01:00+00:00',
                    'local_time_offset_minute': 120,
                    'response_time': 1000,
                },
                'usage': {
                    'input_tokens': 10,
                    'output_tokens': 20,
                    'total_tokens': 30,
                    'estimated_cost': 5,
                },
            },
            'connection': {
                'result_source': 'PROVIDER',
                'cache_look_fail_reason': 'CACHE_NOT_FOUND',
                'caller_type': 'SDK',
                'caller_app': 'PYTHON_SDK',
            },
        }
    )
    _verify_log_messages(temp_dir, [])

  def test_upload_with_network_error(self, requests_mock):
    """Tests handling of network errors during upload."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    call_record = _create_test_call_record()
    test_cases = [
        (requests.exceptions.ConnectionError, 'Connection error'),
        (requests.exceptions.Timeout, 'Request timed out'),
        (requests.exceptions.RequestException, 'General request error'),
    ]
    for _idx, (exception_class, error_message) in enumerate(test_cases):
      if os.path.exists(os.path.join(temp_dir, 'merged.log')):
        os.remove(os.path.join(temp_dir, 'merged.log'))
      requests_mock.post(
          'https://proxainest-production.up.railway.app/ingestion/call-records',
          exc=exception_class(error_message)
      )
      connection.upload_call_record(call_record)
      _verify_log_messages(
          temp_dir, [(
              f'ProxDash could not log the record. Error: {error_message}',
              types.LoggingType.ERROR
          )]
      )

  def test_upload_with_invalid_api_key(self, requests_mock):
    """Tests behavior when API key is invalid or expired."""
    connection, temp_dir, temp_dir_obj = _create_connection()
    connection.status = types.ProxDashConnectionStatus.API_KEY_NOT_VALID
    call_record = _create_test_call_record()
    connection.upload_call_record(call_record)
    _verify_proxdash_request(requests_mock)
    _verify_log_messages(temp_dir, [])


class TestProxDashConnectionGetModelConfigsSchema:

  @pytest.fixture(autouse=True)
  def setup_get_model_configs_schema(self, monkeypatch):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    self.base_url = 'https://proxainest-production.up.railway.app'
    # Create a valid config with 4+ providers for validation
    self.valid_config = json.dumps({
        "metadata": {
            "version": "1.0.0",
            "config_origin": "PROXDASH",
            "release_notes": "TEST CONFIG"
        },
        "version_config": {
            "provider_model_configs": {
                "openai": {
                    "gpt-4": {}
                },
                "claude": {
                    "claude-3": {}
                },
                "gemini": {
                    "gemini-pro": {}
                },
                "mistral": {
                    "mistral-large": {}
                }
            }
        }
    })
    yield

  def _create_connection(
      self, connected: bool = True, requests_mock=None
  ) -> proxdash.ProxDashConnection:
    if connected and requests_mock:
      requests_mock.get(
          f'{self.base_url}/ingestion/verify-key',
          text='{"success": true, "data": {"permission": "ALL"}}',
          status_code=200
      )
    connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(
            api_key='test_api_key' if connected else None,
            disable_proxdash=not connected
        )
    )
    return proxdash.ProxDashConnection(init_from_params=connection_params)

  def test_successful_fetch_when_connected(self, requests_mock):
    """Tests successful fetch with API key authentication."""
    requests_mock.get(
        f'{self.base_url}/ingestion/verify-key',
        text='{"success": true, "data": {"permission": "ALL"}}', status_code=200
    )
    connection = self._create_connection(
        connected=True, requests_mock=requests_mock
    )

    requests_mock.get(
        requests_mock_module.ANY,
        text='{"success": true, "data": ' + self.valid_config + '}',
        status_code=200
    )

    result = connection.get_model_configs_schema()

    assert result is not None
    assert result.metadata.config_origin == types.ConfigOriginType.PROXDASH
    assert result.metadata.release_notes == 'TEST CONFIG'

  def test_successful_fetch_when_not_connected(self, requests_mock):
    """Tests successful fetch without API key (public endpoint)."""
    connection = self._create_connection(connected=False)

    requests_mock.get(
        requests_mock_module.ANY,
        text='{"success": true, "data": ' + self.valid_config + '}',
        status_code=200
    )

    result = connection.get_model_configs_schema()

    assert result is not None
    assert result.metadata.config_origin == types.ConfigOriginType.PROXDASH

  def test_returns_none_on_non_200_status(self, requests_mock):
    """Tests that None is returned when API returns non-200 status."""
    connection = self._create_connection(connected=False)

    requests_mock.get(
        requests_mock_module.ANY, text='Server Error', status_code=500
    )

    result = connection.get_model_configs_schema()

    assert result is None

  def test_returns_none_on_unsuccessful_response(self, requests_mock):
    """Tests that None is returned when success is false."""
    connection = self._create_connection(connected=False)

    requests_mock.get(
        requests_mock_module.ANY,
        text='{"success": false, "error": "Some error"}', status_code=200
    )

    result = connection.get_model_configs_schema()

    assert result is None

  def test_returns_none_on_decode_error(self, requests_mock):
    """Tests that None is returned when response data cannot be decoded."""
    connection = self._create_connection(connected=False)

    # Use invalid enum value for config_origin to trigger decode error
    invalid_config = '{"metadata": {"config_origin": "INVALID_ORIGIN"}}'
    requests_mock.get(
        requests_mock_module.ANY,
        text='{"success": true, "data": ' + invalid_config + '}',
        status_code=200
    )

    result = connection.get_model_configs_schema()

    assert result is None

  def test_returns_none_on_insufficient_providers(self, requests_mock):
    """Tests that None is returned when schema has fewer than 4 providers."""
    connection = self._create_connection(connected=False)

    # Config with only 2 providers (less than required 4)
    insufficient_config = json.dumps({
        "metadata": {
            "config_origin": "PROXDASH"
        },
        "version_config": {
            "provider_model_configs": {
                "openai": {
                    "gpt-4": {}
                },
            }
        }
    })
    requests_mock.get(
        requests_mock_module.ANY,
        text='{"success": true, "data": ' + insufficient_config + '}',
        status_code=200
    )

    result = connection.get_model_configs_schema()

    assert result is None


class TestProxDashConnectionGetProviderApiKeys:

  @pytest.fixture(autouse=True)
  def setup_get_provider_api_keys(self, monkeypatch):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    self.base_url = 'https://proxainest-production.up.railway.app'
    yield

  def _create_connection(
      self, connected: bool = True, requests_mock=None
  ) -> proxdash.ProxDashConnection:
    if connected and requests_mock:
      requests_mock.get(
          f'{self.base_url}/ingestion/verify-key',
          text='{"success": true, "data": {"permission": "ALL"}}',
          status_code=200
      )
    connection_params = proxdash.ProxDashConnectionParams(
        logging_options=types.LoggingOptions(),
        proxdash_options=types.ProxDashOptions(
            api_key='test_api_key' if connected else None,
            disable_proxdash=not connected
        )
    )
    return proxdash.ProxDashConnection(init_from_params=connection_params)

  def test_returns_empty_dict_when_not_connected(self, requests_mock):
    """Tests that empty dict is returned when connection status is not CONNECTED."""
    connection = self._create_connection(connected=False)

    result = connection.get_provider_api_keys()

    assert result == {}

  def test_successful_fetch_returns_provider_keys(self, requests_mock):
    """Tests successful fetch returns provider API keys."""
    connection = self._create_connection(
        connected=True, requests_mock=requests_mock
    )

    expected_keys = {
        'OPENAI_API_KEY': 'sk-test-openai-key',
        'ANTHROPIC_API_KEY': 'sk-test-anthropic-key'
    }
    requests_mock.get(
        f'{self.base_url}/ingestion/provider-connection-keys', text=json.dumps({
            'success': True,
            'data': expected_keys
        }), status_code=200
    )

    result = connection.get_provider_api_keys()

    assert result == expected_keys

  def test_returns_empty_dict_on_non_200_status(self, requests_mock):
    """Tests that empty dict is returned when API returns non-200 status."""
    connection = self._create_connection(
        connected=True, requests_mock=requests_mock
    )

    requests_mock.get(
        f'{self.base_url}/ingestion/provider-connection-keys',
        text='Server Error', status_code=500
    )

    result = connection.get_provider_api_keys()

    assert result == {}

  def test_returns_empty_dict_on_unsuccessful_response(self, requests_mock):
    """Tests that empty dict is returned when success is false."""
    connection = self._create_connection(
        connected=True, requests_mock=requests_mock
    )

    requests_mock.get(
        f'{self.base_url}/ingestion/provider-connection-keys',
        text='{"success": false, "error": "Some error"}', status_code=200
    )

    result = connection.get_provider_api_keys()

    assert result == {}

  def test_request_includes_api_key_header(self, requests_mock):
    """Tests that API key is included in request header."""
    connection = self._create_connection(
        connected=True, requests_mock=requests_mock
    )

    requests_mock.get(
        f'{self.base_url}/ingestion/provider-connection-keys',
        text='{"success": true, "data": {}}', status_code=200
    )

    connection.get_provider_api_keys()

    # Find the request to provider-connection-keys
    provider_key_requests = [
        req for req in requests_mock.request_history
        if 'provider-connection-keys' in req.url
    ]
    assert len(provider_key_requests) == 1
    assert provider_key_requests[0].headers['X-API-Key'] == 'test_api_key'


class TestFileInfoToMessageContent:

  def test_converts_backend_response_to_message_content(self):
    """Backend FileInfo (camelCase) converts to MessageContent correctly."""
    connection, _, _ = _create_connection()
    info = {
        'id': 'pd-123',
        'type': 'image',
        'filename': 'cat.jpeg',
        'mimeType': 'image/jpeg',
        'sizeBytes': 5000,
        'uploadConfirmed': True,
        'providerFileApiIds': {
            'gemini': 'files/g-1',
            'openai': 'file-o-1',
        },
        'providerFileApiStatus': {
            'gemini': {
                'file_id': 'files/g-1',
                'provider': 'gemini',
                'state': 'active',
            },
        },
        'source': 'https://s3.example.com/presigned-url',
        'createdAt': '2024-01-01T00:00:00Z',
        'updatedAt': '2024-01-01T00:01:00Z',
    }
    mc = connection._file_info_to_message_content(info)
    assert mc is not None
    assert mc.type == message_content.ContentType.IMAGE
    assert mc.media_type == 'image/jpeg'
    assert mc.filename == 'cat.jpeg'
    assert mc.source == 'https://s3.example.com/presigned-url'
    assert mc.proxdash_file_id == 'pd-123'
    assert mc.proxdash_file_status.file_id == 'pd-123'
    assert mc.proxdash_file_status.upload_confirmed is True
    assert mc.provider_file_api_ids == {
        'gemini': 'files/g-1',
        'openai': 'file-o-1',
    }
    assert mc.provider_file_api_status['gemini'].file_id == 'files/g-1'
    assert mc.provider_file_api_status['gemini'].state == (
        message_content.FileUploadState.ACTIVE
    )


class TestUploadCallRecordMediaHandling:

  def test_media_uploaded_and_bytes_stripped(self, requests_mock):
    """Media content is uploaded to ProxDash file API before the call
    record is sent, and data/path are stripped from the payload."""
    connection, temp_dir, temp_dir_obj = _create_connection()

    # Build a call record with an image in query chat.
    img = message_content.MessageContent(
        type=message_content.ContentType.IMAGE,
        data=b'\x89PNG-fake-image-bytes',
        media_type='image/png',
    )
    model_configs_instance = model_configs.ModelConfigs()
    call_record = types.CallRecord(
        query=types.QueryRecord(
            chat=chat_session.Chat(messages=[
                message.Message(
                    role='user',
                    content=[
                        img,
                        message_content.MessageContent(
                            type='text', text='What is this?'),
                    ])
            ]),
            provider_model=model_configs_instance.get_provider_model(
                ('mock_provider', 'mock_model')),
        ),
        result=types.ResultRecord(
            status=types.ResultStatusType.SUCCESS,
            output_text='A cat.',
            timestamp=types.TimeStampType(
                start_utc_date=datetime.datetime(2024, 1, 1, 12, 0),
                end_utc_date=datetime.datetime(2024, 1, 1, 12, 1),
                response_time=datetime.timedelta(seconds=1),
            ),
        ),
        connection=types.ConnectionMetadata(
            result_source=types.ResultSource.PROVIDER),
    )

    # Mock ProxDash file upload (3 requests: upload, S3 PUT, confirm).
    requests_mock.post(
        'https://proxainest-production.up.railway.app/files/upload',
        json={
            'success': True,
            'data': {
                'id': 'pd-img-1',
                'presignedUploadUrl': 'https://s3.example.com/put-here',
                's3Key': 'files/user1/pd-img-1',
            }
        },
        status_code=201,
    )
    requests_mock.put('https://s3.example.com/put-here', status_code=200)
    requests_mock.post(
        'https://proxainest-production.up.railway.app/files/update/pd-img-1',
        json={'success': True},
        status_code=200,
    )
    # Mock call record upload.
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        json={'success': True},
        status_code=201,
    )

    connection.upload_call_record(call_record)

    # Verify: image got a proxdash_file_id.
    assert img.proxdash_file_id == 'pd-img-1'

    # Verify: call record payload has no data/path on the image block.
    call_record_requests = [
        r for r in requests_mock.request_history
        if 'call-records' in r.url
    ]
    assert len(call_record_requests) == 1
    payload = json.loads(call_record_requests[0].text)
    chat_messages = payload['query']['chat']['messages']
    img_block = [
        c for c in chat_messages[0]['content']
        if c.get('type') == 'image'
    ][0]
    assert 'data' not in img_block
    assert 'path' not in img_block
    assert img_block.get('proxdash_file_id') == 'pd-img-1'


class TestProxDashConnectionUploadCallRecordResponseTypes:
  """Tests for upload_call_record with different response types."""

  def test_upload_with_json_response(self, requests_mock):
    """Tests upload with JSON output."""
    connection, temp_dir, _ = _create_connection()
    call_record = _create_test_call_record(
        output_text=None,
        output_json={
            'result': 'success',
            'count': 5
        },
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock,
        {'result': {
            'output_json': {
                'result': 'success',
                'count': 5
            },
        }}
    )

  def test_upload_with_pydantic_response(self, requests_mock):
    """Tests upload with pydantic output."""
    connection, temp_dir, _ = _create_connection()
    pydantic_instance = SamplePydanticModel(name='test_name', value=789)
    call_record = _create_test_call_record(
        output_text=None,
        output_pydantic=pydantic_instance,
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=SamplePydanticModel,
        ),
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'result': {
                'output_pydantic': {
                    'class_name': 'SamplePydanticModel',
                    'instance_json_value': {
                        'name': 'test_name',
                        'value': 789,
                        'active': True,
                    },
                },
            },
            'query': {
                'output_format': {
                    'type': 'PYDANTIC',
                    'pydantic_class_name': 'SamplePydanticModel',
                },
            },
        }
    )

  def test_upload_with_pydantic_hidden_content(self, requests_mock):
    """Tests upload with pydantic output when sensitive content is hidden."""
    connection, temp_dir, _ = _create_connection(hide_sensitive_content=True)
    pydantic_instance = SamplePydanticModel(name='secret', value=999)
    call_record = _create_test_call_record(
        output_text='sensitive response',
        output_pydantic=pydantic_instance,
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=SamplePydanticModel,
        ),
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    hidden_str = proxdash._SENSITIVE_CONTENT_HIDDEN_STRING
    actual_data = json.loads([
        r for r in requests_mock.request_history if 'call-records' in r.url
    ][0].text)
    assert actual_data['query']['prompt'] == hidden_str
    assert actual_data['query']['system_prompt'] == hidden_str
    assert actual_data['result']['output_text'] == hidden_str
    assert 'output_pydantic' not in actual_data['result']

  def test_upload_with_json_output_format(self, requests_mock):
    """Tests upload with JSON output_format."""
    connection, temp_dir, _ = _create_connection()
    call_record = _create_test_call_record(
        output_text=None,
        output_json={'data': 'value'},
        output_format=types.OutputFormat(type=types.OutputFormatType.JSON),
    )
    requests_mock.post(
        'https://proxainest-production.up.railway.app/ingestion/call-records',
        text='{"success": true}', status_code=201
    )
    connection.upload_call_record(call_record)
    _verify_proxdash_request(
        requests_mock, {
            'result': {
                'output_json': {
                    'data': 'value'
                },
            },
            'query': {
                'output_format': {
                    'type': 'JSON',
                },
            },
        }
    )

import os
import json
import tempfile
import pytest
import proxai.types as types
from proxai.connections.proxdash import ProxDashConnection
from proxai.logging.utils import log_proxdash_message
import requests


class TestProxDashConnectionInit:
  @pytest.fixture(autouse=True)
  def setup_test(self, monkeypatch, requests_mock):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='true',
        status_code=201,
    )
    yield

  def test_init_disabled(self):
    connection = ProxDashConnection(
        proxdash_options=types.ProxDashOptions(disable_proxdash=True))
    assert connection.status == types.ProxDashConnectionStatus.DISABLED

  def test_init_no_api_key(self):
    connection = ProxDashConnection()
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_FOUND)

  def test_init_with_invalid_env_api_key(self, monkeypatch, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='false',
        status_code=201,
    )
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_api_key')
    connection = ProxDashConnection()
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_VALID)
    assert len(requests_mock.request_history) == 1

  def test_init_with_incorrect_proxdash_response(
      self, monkeypatch, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='Some invalid response',
        status_code=201,
    )
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_api_key')
    connection = ProxDashConnection()
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.PROXDASH_INVALID_RETURN)
    assert len(requests_mock.request_history) == 1

  def test_init_with_valid_env_api_key(self, monkeypatch, requests_mock):
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_api_key')
    connection = ProxDashConnection()
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert len(requests_mock.request_history) == 1

  def test_init_with_invalid_api_key(self, monkeypatch, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='false',
        status_code=201,
    )
    connection = ProxDashConnection(api_key='invalid_api_key')
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.API_KEY_NOT_VALID)
    assert len(requests_mock.request_history) == 1

  def test_init_with_valid_api_key(self, monkeypatch, requests_mock):
    connection = ProxDashConnection(api_key='test_api_key')
    assert (
        connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert len(requests_mock.request_history) == 1

  def test_init_with_invalid_combinations(self):
    with pytest.raises(ValueError):
      ProxDashConnection(
          experiment_path='path',
          get_experiment_path=lambda: 'path')

    with pytest.raises(ValueError):
      ProxDashConnection(
          logging_options=types.LoggingOptions(),
          get_logging_options=lambda: types.LoggingOptions())

    with pytest.raises(ValueError):
      ProxDashConnection(
          proxdash_options=types.ProxDashOptions(),
          get_proxdash_options=lambda: types.ProxDashOptions())


class TestProxDashConnectionInitState:
  @pytest.fixture(autouse=True)
  def setup_test(self, monkeypatch, requests_mock):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='true',
        status_code=201,
    )
    yield

  def test_simple_init_state(self):
    init_state = types.ProxDashInitState(
        status=types.ProxDashConnectionStatus.INITIALIZING,
        hidden_run_key='test_key',
        api_key='test_api_key',
        experiment_path='test/path')
    connection = ProxDashConnection(init_state=init_state)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED

  def test_validation_skipped_if_init_state_already_validated(
      self, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='true',
        status_code=201,
    )
    init_state = types.ProxDashInitState(
        status=types.ProxDashConnectionStatus.CONNECTED,
        hidden_run_key='test_key',
        api_key='test_api_key',
        experiment_path='test/path')
    connection = ProxDashConnection(init_state=init_state)
    assert connection.status == types.ProxDashConnectionStatus.CONNECTED

  def test_get_init_state(self):
    base_logging_options = types.LoggingOptions(stdout=True)
    base_proxdash_options = types.ProxDashOptions(stdout=True)
    connection = ProxDashConnection(
        hidden_run_key='test_key',
        api_key='test_api_key',
        experiment_path='test/path',
        logging_options=base_logging_options,
        proxdash_options=base_proxdash_options)

    init_state = connection.get_init_state()
    assert (
        init_state.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert init_state.hidden_run_key == 'test_key'
    assert init_state.api_key == 'test_api_key'
    assert init_state.experiment_path == 'test/path'
    assert init_state.logging_options.stdout == True
    assert init_state.proxdash_options.stdout == True

  def test_init_with_init_state(self):
    # First create a connection and get its state
    base_logging_options = types.LoggingOptions(stdout=True)
    base_proxdash_options = types.ProxDashOptions(stdout=True)
    original_connection = ProxDashConnection(
        hidden_run_key='test_key',
        api_key='test_api_key',
        experiment_path='test/path',
        logging_options=base_logging_options,
        proxdash_options=base_proxdash_options)

    assert (
        original_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)

    init_state = original_connection.get_init_state()
    new_connection = ProxDashConnection(init_state=init_state)

    assert new_connection.status == original_connection.status
    assert (
        new_connection._hidden_run_key ==
        original_connection._hidden_run_key)
    assert new_connection._api_key == original_connection._api_key
    assert (
        new_connection.experiment_path ==
        original_connection.experiment_path)
    assert (
        new_connection.logging_options.stdout ==
        original_connection.logging_options.stdout)
    assert (
        new_connection.proxdash_options.stdout ==
        original_connection.proxdash_options.stdout)

  def test_init_state_invalid_combinations(self):
    init_state = types.ProxDashInitState(
        status=types.ProxDashConnectionStatus.API_KEY_NOT_VALID,
        hidden_run_key='test_key')

    with pytest.raises(ValueError):
      ProxDashConnection(
          init_state=init_state,
          hidden_run_key='another_key')

    with pytest.raises(ValueError):
      ProxDashConnection(
          init_state=init_state,
          api_key='test_api_key')

    with pytest.raises(ValueError):
      ProxDashConnection(
          init_state=init_state,
          experiment_path='test/path')

  def test_init_state_with_all_options(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      init_state = types.ProxDashInitState(
          status=types.ProxDashConnectionStatus.CONNECTED,
          hidden_run_key='test_key',
          api_key='test_api_key',
          experiment_path='test/path',
          logging_options=types.LoggingOptions(
              stdout=True,
              hide_sensitive_content=True,
              logging_path=temp_dir
          ),
          proxdash_options=types.ProxDashOptions(
              stdout=True,
              hide_sensitive_content=True,
              disable_proxdash=False
          ),
      )

      connection = ProxDashConnection(init_state=init_state)

      assert connection.status == types.ProxDashConnectionStatus.CONNECTED
      assert connection._hidden_run_key == 'test_key'
      assert connection._api_key == 'test_api_key'
      assert connection.experiment_path == 'test/path'
      assert connection.logging_options.stdout == True
      assert connection.logging_options.hide_sensitive_content == True
      assert connection.logging_options.logging_path == temp_dir
      assert connection.proxdash_options.stdout == True
      assert connection.proxdash_options.hide_sensitive_content == True
      assert connection.proxdash_options.disable_proxdash == False


class TestProxDashConnectionProperties:
  @pytest.fixture(autouse=True)
  def setup_test(self, monkeypatch, requests_mock):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    requests_mock.post(
        'https://proxainest-production.up.railway.app/connect',
        text='true',
        status_code=201,
    )
    yield

  def test_init_with_get_functions(self):
    def get_logging_options():
      return types.LoggingOptions(stdout=True)

    def get_proxdash_options():
      return types.ProxDashOptions(stdout=True)

    def get_experiment_path():
      return 'dynamic/test/path'

    connection = ProxDashConnection(
        get_logging_options=get_logging_options,
        get_proxdash_options=get_proxdash_options,
        get_experiment_path=get_experiment_path
    )

    assert connection.logging_options.stdout == True
    assert connection.proxdash_options.stdout == True
    assert connection.experiment_path == 'dynamic/test/path'

  def test_logging_options_property(self):
    connection = ProxDashConnection(
        logging_options=types.LoggingOptions(stdout=True))
    assert connection.logging_options.stdout == True

  def test_invalid_logging_path(self):
    with pytest.raises(
        FileNotFoundError, match=' No such file or directory'):
      ProxDashConnection(
          logging_options=types.LoggingOptions(
              logging_path='/nonexistent/path'
          ))

  def test_proxdash_options_property(self):
    connection = ProxDashConnection(
        proxdash_options=types.ProxDashOptions(stdout=True))
    assert connection.proxdash_options.stdout == True

  def test_hide_sensitive_content(self):
    connection = ProxDashConnection()
    logging_record = types.LoggingRecord(
        query_record=types.QueryRecord(
            prompt="sensitive prompt",
            system="sensitive system",
            messages=[{"role": "user", "content": "sensitive message"}]
        ),
        response_record=types.QueryResponseRecord(
            response="sensitive response"
        ),
        response_source=types.ResponseSource.PROVIDER
    )

    hidden_record = connection._hide_sensitive_content_logging_record(
        logging_record)
    assert hidden_record.query_record.prompt == '<sensitive content hidden>'
    assert hidden_record.query_record.system == '<sensitive content hidden>'
    assert hidden_record.query_record.messages == [
        {'role': 'assistant', 'content': '<sensitive content hidden>'}
    ]
    assert (
        hidden_record.response_record.response ==
        '<sensitive content hidden>')

  def test_log_proxdash_message(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      logging_options = types.LoggingOptions(
          logging_path=temp_dir,
          hide_sensitive_content=True)
      proxdash_options = types.ProxDashOptions()

      test_message = "Test proxdash message"
      query_record = types.QueryRecord(
          prompt="test prompt",
          model=('test_provider', 'test_model'))

      log_proxdash_message(
          logging_options=logging_options,
          proxdash_options=proxdash_options,
          message=test_message,
          type=types.LoggingType.ERROR,
          query_record=query_record)

      assert os.path.exists(os.path.join(temp_dir, 'errors.log'))
      assert os.path.exists(os.path.join(temp_dir, 'proxdash.log'))

  def test_experiment_path_setter(self):
    connection = ProxDashConnection(api_key='test_api_key')
    connection.experiment_path = 'test/path'
    assert connection.experiment_path == 'test/path'

  def test_experiment_path_getter_function(self):
    def get_path():
        return 'dynamic/path'

    connection = ProxDashConnection(
        api_key='test_api_key',
        get_experiment_path=get_path)
    assert connection.experiment_path == 'dynamic/path'

  def test_invalid_experiment_path(self):
    with pytest.raises(
        ValueError, match='Experiment path cannot start with "/"'):
      ProxDashConnection(experiment_path='///invalid')

  def test_experiment_path_updates(self):
    connection = ProxDashConnection(api_key='test_api_key')

    assert connection._last_connected_experiment_path == '(not set)'
    assert connection._experiment_path == None
    assert connection.experiment_path == '(not set)'

    connection.experiment_path = 'initial/path'

    assert connection._last_connected_experiment_path == 'initial/path'
    assert connection._experiment_path == 'initial/path'
    assert connection.experiment_path == 'initial/path'

    connection = ProxDashConnection(
        api_key='test_api_key',
        experiment_path='initial/path_2')

    assert connection._last_connected_experiment_path == 'initial/path_2'
    assert connection._experiment_path == 'initial/path_2'
    assert connection.experiment_path == 'initial/path_2'

    connection.experiment_path = 'new/path'
    assert connection._last_connected_experiment_path == 'new/path'
    assert connection._experiment_path == 'new/path'
    assert connection.experiment_path == 'new/path'

  def test_experiment_path_logging(self):
    def _get_proxdash_log_message(path):
      result = []
      assert os.path.exists(os.path.join(path, 'proxdash.log'))
      with open(os.path.join(path, 'proxdash.log'), 'r') as f:
        for line in f:
          data = json.loads(line)
          if 'Connected to ProxDash experiment:' in data['message']:
            result.append(data['message'])
      return result

    with tempfile.TemporaryDirectory() as temp_dir:
      connection = ProxDashConnection(
        api_key='test_api_key',
        logging_options=types.LoggingOptions(
            logging_path=temp_dir))

      assert _get_proxdash_log_message(temp_dir) == [
        'Connected to ProxDash experiment: (not set)'
      ]

      connection.experiment_path = 'initial/path'
      assert _get_proxdash_log_message(temp_dir) == [
        'Connected to ProxDash experiment: (not set)',
        'Connected to ProxDash experiment: initial/path'
      ]

      connection = ProxDashConnection(
        api_key='test_api_key',
        experiment_path='initial/path_2',
        logging_options=types.LoggingOptions(
            logging_path=temp_dir))

      assert _get_proxdash_log_message(temp_dir) == [
        'Connected to ProxDash experiment: (not set)',
        'Connected to ProxDash experiment: initial/path',
        'Connected to ProxDash experiment: initial/path_2'
      ]

      connection.experiment_path = 'new/path'
      assert _get_proxdash_log_message(temp_dir) == [
        'Connected to ProxDash experiment: (not set)',
        'Connected to ProxDash experiment: initial/path',
        'Connected to ProxDash experiment: initial/path_2',
        'Connected to ProxDash experiment: new/path'
      ]


  def test_upload_logging_record_disabled(self, requests_mock):
    requests_mock.post(
        'https://proxainest-production.up.railway.app/logging-record',
        text='success',
        status_code=201,
    )
    proxdash_options = types.ProxDashOptions(disable_proxdash=True)
    connection = ProxDashConnection(
        proxdash_options=proxdash_options,
        hidden_run_key='test_key')

    logging_record = types.LoggingRecord(
        query_record=types.QueryRecord(
            model=('test_provider', 'test_model')),
        response_record=types.QueryResponseRecord(),
        response_source=types.ResponseSource.PROVIDER)

    connection.upload_logging_record(logging_record)
    assert requests_mock.call_count == 0

import pytest
import datetime
import tempfile
import copy
from typing import Dict, Optional
import proxai.types as types
import proxai.caching.query_cache as query_cache
import proxai.stat_types as stats_type
import proxai.connections.proxdash as proxdash
import proxai.connectors.providers.mock_provider as mock_provider
import proxai.connectors.model_configs as model_configs


@pytest.fixture(autouse=True)
def setup_test(monkeypatch, requests_mock):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.setenv(api_key, 'test_api_key')
  requests_mock.post(
      'https://proxainest-production.up.railway.app/connect',
      text='{"permission": "ALL"}',
      status_code=201,
  )
  yield


def get_mock_provider_model_connector(
    strict_feature_test: Optional[bool] = None,
    query_cache_manager: Optional[query_cache.QueryCacheManager] = None,
    stats: Optional[Dict[str, stats_type.ProviderModelStats]] = None,
):
  connector = mock_provider.MockProviderModelConnector(
      provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
      run_type=types.RunType.TEST,
      logging_options=types.LoggingOptions(),
      strict_feature_test=strict_feature_test,
      query_cache_manager=query_cache_manager,
      stats=stats,
      proxdash_connection=proxdash.ProxDashConnection(
          init_state=types.ProxDashConnectionState(
              status=types.ProxDashConnectionStatus.CONNECTED,
              api_key='test_api_key',
              experiment_path='test/path',
              logging_options=types.LoggingOptions(),
              proxdash_options=types.ProxDashOptions(),
              key_info_from_proxdash={'permission': 'ALL'},
              connected_experiment_path='test/path')))
  return connector


class TestModelConnectorGettersSetters:
  def test_api_property(self):
    connector = get_mock_provider_model_connector()
    assert connector.api is None  # Based on our mock implementation

    # Test caching behavior
    connector._api = "test_api"
    assert connector.api == "test_api"

    with pytest.raises(
        ValueError,
        match='api should not be set directly.'):
      connector.api = "test_api_2"

  def test_generic_property(self):
    connector = get_mock_provider_model_connector()
    assert connector.provider_model == model_configs.ALL_MODELS[
        'mock_provider']['mock_model']
    assert (
        connector._provider_model_state.provider_model ==
        model_configs.ALL_MODELS['mock_provider']['mock_model'])

    connector.provider_model = 'test_provider_model'
    assert connector.provider_model == 'test_provider_model'
    assert (
        connector._provider_model_state.provider_model == 'test_provider_model')

  def test_proxdash_connection(self):
    connector = get_mock_provider_model_connector()
    assert isinstance(
        connector.proxdash_connection, proxdash.ProxDashConnection)
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert connector.proxdash_connection.api_key == 'test_api_key'
    assert connector.proxdash_connection.experiment_path == 'test/path'

    assert isinstance(
        connector._provider_model_state.proxdash_connection,
        types.ProxDashConnectionState)
    assert (
        connector._provider_model_state.proxdash_connection ==
        connector.proxdash_connection.get_state())

  def test_proxdash_connection_function(self):
    dynamic_proxdash_connection = proxdash.ProxDashConnection(
        init_state=types.ProxDashConnectionState(
            status=types.ProxDashConnectionStatus.CONNECTED,
            api_key='test_api_key',
            experiment_path='test/path',
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(),
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

    def get_proxdash_connection():
      return dynamic_proxdash_connection

    connector = mock_provider.MockProviderModelConnector(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST,
        logging_options=types.LoggingOptions(),
        get_proxdash_connection=get_proxdash_connection)

    assert isinstance(
        connector.proxdash_connection, proxdash.ProxDashConnection)
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert connector.proxdash_connection.experiment_path == 'test/path'
    assert (
        connector._provider_model_state.proxdash_connection ==
        dynamic_proxdash_connection.get_state())

    first_dynamic_proxdash_connection = copy.deepcopy(
        dynamic_proxdash_connection)
    dynamic_proxdash_connection.experiment_path = 'test/path_2'
    assert (
        connector._provider_model_state.proxdash_connection ==
        first_dynamic_proxdash_connection.get_state())
    # This getter updates the proxdash connection state:
    assert (
        connector.proxdash_connection.experiment_path ==
        'test/path_2')
    assert (
        connector._provider_model_state.proxdash_connection ==
        dynamic_proxdash_connection.get_state())


class TestModelConnectorInit:
  def test_init_state(self):
    init_state = types.ProviderModelState(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST,
        strict_feature_test=True,
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_connection=types.ProxDashConnectionState(
            status=types.ProxDashConnectionStatus.CONNECTED,
            hidden_run_key='test_key',
            experiment_path='test/path',
            api_key='test_api_key',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(),
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

    connector = mock_provider.MockProviderModelConnector(init_state=init_state)

    assert connector.provider_model == model_configs.ALL_MODELS[
        'mock_provider']['mock_model']
    assert connector.run_type == types.RunType.TEST
    assert connector.strict_feature_test == True
    assert connector.logging_options.stdout == True
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert connector.proxdash_connection.api_key == 'test_api_key'

  def test_init_with_mismatched_model(self):
    init_state = types.ProviderModelState(
        provider_model=model_configs.ALL_MODELS['claude']['claude-3-opus'],
        run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match=(
            'provider_model needs to be same with the class provider name.\n'
            'provider_model: *')):
      mock_provider.MockProviderModelConnector(init_state=init_state)

  def test_init_state_with_all_options(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      init_state = types.ProviderModelState(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          strict_feature_test=True,
          logging_options=types.LoggingOptions(
              logging_path=temp_dir,
              hide_sensitive_content=True,
              stdout=True),
          proxdash_connection=types.ProxDashConnectionState(
              status=types.ProxDashConnectionStatus.CONNECTED,
              hidden_run_key='test_key',
              api_key='test_api_key',
              experiment_path='test/path',
              logging_options=types.LoggingOptions(
                  logging_path=temp_dir,
                  hide_sensitive_content=True,
                  stdout=True),
              proxdash_options=types.ProxDashOptions(stdout=True)))

      connector = mock_provider.MockProviderModelConnector(
          init_state=init_state)

      assert connector.provider_model == model_configs.ALL_MODELS[
          'mock_provider']['mock_model']
      assert connector.run_type == types.RunType.TEST
      assert connector.strict_feature_test == True
      assert connector.logging_options.stdout == True
      assert connector.logging_options.hide_sensitive_content == True
      assert connector.logging_options.logging_path == temp_dir
      assert (
          connector.proxdash_connection.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert connector.proxdash_connection._hidden_run_key == 'test_key'
      assert connector.proxdash_connection._api_key == 'test_api_key'
      assert connector.proxdash_connection.experiment_path == 'test/path'

  def test_init_with_literals(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      base_logging_options = types.LoggingOptions(
          stdout=True,
          hide_sensitive_content=True,
          logging_path=temp_dir)

      proxdash_connection = proxdash.ProxDashConnection(
          hidden_run_key='test_key',
          api_key='test_api_key',
          experiment_path='test/path',
          logging_options=base_logging_options,
          proxdash_options=types.ProxDashOptions(stdout=True))

      connector = mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          strict_feature_test=True,
          logging_options=base_logging_options,
          proxdash_connection=proxdash_connection)

      init_state = connector.get_state()
      assert init_state.provider_model == model_configs.ALL_MODELS[
          'mock_provider']['mock_model']
      assert init_state.run_type == types.RunType.TEST
      assert init_state.strict_feature_test == True
      assert init_state.logging_options.stdout == True
      assert init_state.logging_options.hide_sensitive_content == True
      assert init_state.logging_options.logging_path == temp_dir
      assert (
          init_state.proxdash_connection.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert init_state.proxdash_connection.hidden_run_key == 'test_key'
      assert init_state.proxdash_connection.api_key == 'test_api_key'
      assert init_state.proxdash_connection.experiment_path == 'test/path'

  def test_init_with_none_model(self):
    init_state = types.ProviderModelState(run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match='provider_model needs to be set in init_state.'):
      mock_provider.MockProviderModelConnector(init_state=init_state)

  def test_init_with_invalid_combinations(self):
    with pytest.raises(
        ValueError,
        match=(
            'Only one of logging_options or get_logging_options should be set '
            'while initializing the StateControlled object.')):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          logging_options=types.LoggingOptions(stdout=True),
          get_logging_options=lambda: types.LoggingOptions(stdout=True))

  def test_invalid_model_combination(self):
    connector = get_mock_provider_model_connector()
    with pytest.raises(
        ValueError,
        match=(
            'provider_model does not match the connector provider_model.'
            'provider_model: *')):
      connector.generate_text(
          prompt="Hello",
          provider_model=model_configs.ALL_MODELS['claude']['claude-3-opus'])


class TestModelConnector:
  def test_mutually_exclusive_params(self):
    connector = get_mock_provider_model_connector()
    with pytest.raises(
        ValueError,
        match="prompt and messages cannot be set at the same time"):
      connector.generate_text(
          prompt="Hello",
          messages=[{"role": "user", "content": "Hello"}])

  def test_strict_feature_test_true(self):
    connector = get_mock_provider_model_connector(strict_feature_test=True)
    with pytest.raises(Exception, match="Test error"):
      connector.feature_fail("Test error")

  def test_strict_feature_test_false(self):
    connector = get_mock_provider_model_connector(strict_feature_test=False)
    # Should not raise an exception in non-strict mode
    connector.feature_fail("Test error")

  def test_generate_text(self):
    connector = get_mock_provider_model_connector()
    result = connector.generate_text(
        prompt="Hello",
        max_tokens=100)

    assert isinstance(result, types.LoggingRecord)
    assert result.response_record.response == "mock response"
    assert result.query_record.prompt == "Hello"
    assert result.query_record.max_tokens == 100
    assert result.response_source == types.ResponseSource.PROVIDER

  def test_generate_text_with_cache(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      cache_manager = query_cache.QueryCacheManager(
          cache_options=types.CacheOptions(cache_path=temp_dir))

      connector = get_mock_provider_model_connector(
          query_cache_manager=cache_manager)

      # First call - should hit the provider
      result1 = connector.generate_text(prompt="Hello")
      assert result1.response_source == types.ResponseSource.PROVIDER

      # Second call - should hit the cache
      result2 = connector.generate_text(prompt="Hello")
      assert result2.response_source == types.ResponseSource.CACHE

  def test_stats_update(self):
    provider_model = model_configs.ALL_MODELS['mock_provider']['mock_model']
    stats = {
        stats_type.GlobalStatType.RUN_TIME: stats_type.ProviderModelStats(
            provider_model=provider_model),
        stats_type.GlobalStatType.SINCE_CONNECT: stats_type.ProviderModelStats(
            provider_model=provider_model)
    }

    connector = get_mock_provider_model_connector(stats=stats)

    result = connector.generate_text(prompt="Hello")

    # Verify stats were updated
    run_time_stats = stats[stats_type.GlobalStatType.RUN_TIME]
    assert run_time_stats.provider_stats.total_queries == 1
    assert run_time_stats.provider_stats.total_successes == 1
    assert run_time_stats.provider_stats.total_token_count == 100

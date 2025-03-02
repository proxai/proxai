import pytest
import datetime
import tempfile
from typing import Dict, Optional
import proxai.types as types
import proxai.caching.query_cache as query_cache
import proxai.stat_types as stats_type
import proxai.connections.proxdash as proxdash
import proxai.connectors.mock_provider as mock_provider
import proxai.connectors.model_configs as model_configs


class TestModelConnector:
  @pytest.fixture
  def model_connector(self):
    return mock_provider.MockProviderModelConnector(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST)

  def test_initialization(self, model_connector):
    assert model_connector.provider_model == model_configs.ALL_MODELS[
        'mock_provider']['mock_model']
    assert model_connector.run_type == types.RunType.TEST
    assert model_connector.strict_feature_test == None

  def test_initialization_with_invalid_combinations(self):
    # Both query_cache_manager and get_query_cache_manager are set
    with pytest.raises(ValueError):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          query_cache_manager=query_cache.QueryCacheManager(
              cache_options=types.CacheOptions(cache_path="test_path")),
          get_query_cache_manager=lambda: query_cache.QueryCacheManager(
              cache_options=types.CacheOptions(cache_path="test_path")))

  def test_api_property(self, model_connector):
    assert model_connector.api is None  # Based on our mock implementation

    # Test caching behavior
    model_connector._api = "test_api"
    assert model_connector.api == "test_api"

  def test_feature_fail_strict(self):
    connector = mock_provider.MockProviderModelConnector(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST,
        strict_feature_test=True)

    with pytest.raises(Exception, match="Test error"):
      connector.feature_fail("Test error")

  def test_feature_fail_non_strict(self, model_connector):
    # Should not raise an exception in non-strict mode
    model_connector.feature_fail("Test error")

  def test_generate_text(self, model_connector):
    result = model_connector.generate_text(
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

      connector = mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
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

    connector = mock_provider.MockProviderModelConnector(
        provider_model=provider_model,
        run_type=types.RunType.TEST,
        stats=stats)

    result = connector.generate_text(prompt="Hello")

    # Verify stats were updated
    run_time_stats = stats[stats_type.GlobalStatType.RUN_TIME]
    assert run_time_stats.provider_stats.total_queries == 1
    assert run_time_stats.provider_stats.total_successes == 1
    assert run_time_stats.provider_stats.total_token_count == 100

  def test_invalid_model_combination(self, model_connector):
    with pytest.raises(
        ValueError,
        match=(
            'provider_model does not match the connector provider_model.'
            'provider_model: *')):
      model_connector.generate_text(
          prompt="Hello",
          provider_model=model_configs.ALL_MODELS['claude']['claude-3-opus'])

  def test_mutually_exclusive_params(self, model_connector):
    with pytest.raises(
        ValueError,
        match="prompt and messages cannot be set at the same time"):
      model_connector.generate_text(
          prompt="Hello",
          messages=[{"role": "user", "content": "Hello"}])


class TestModelConnectorInitState:
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
    init_state = types.ModelInitState(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST,
        strict_feature_test=True,
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_init_state=types.ProxDashInitState(
            status=types.ProxDashConnectionStatus.CONNECTED,
            hidden_run_key='test_key',
            api_key='test_api_key'))

    connector = mock_provider.MockProviderModelConnector(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        init_state=init_state)

    assert connector.provider_model == model_configs.ALL_MODELS[
        'mock_provider']['mock_model']
    assert connector.run_type == types.RunType.TEST
    assert connector.strict_feature_test == True
    assert connector.logging_options.stdout == True
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)

  def test_init_with_mismatched_model(self):
    init_state = types.ModelInitState(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match=(
            'init_state.provider_model is not the same as the provider_model '
            'parameter.')):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['claude']['claude-3-opus'],
          init_state=init_state)

  def test_init_with_invalid_combinations(self):
    init_state = types.ModelInitState(
        provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
        run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match='init_state and other parameters cannot be set at the same time'):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          init_state=init_state,
          run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match='init_state and other parameters cannot be set at the same time'):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          init_state=init_state,
          strict_feature_test=True)

  def test_init_with_all_options(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      init_state = types.ModelInitState(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          strict_feature_test=True,
          logging_options=types.LoggingOptions(
              logging_path=temp_dir,
              hide_sensitive_content=True,
              stdout=True),
          proxdash_init_state=types.ProxDashInitState(
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
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
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

  def test_get_init_state(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      base_logging_options = types.LoggingOptions(
          stdout=True,
          hide_sensitive_content=True,
          logging_path=temp_dir)

      proxdash_connection = proxdash.ProxDashConnection(
          hidden_run_key='test_key',
          api_key='test_api_key',
          experiment_path='test/path',
          logging_options=base_logging_options)

      connector = mock_provider.MockProviderModelConnector(
          provider_model=model_configs.ALL_MODELS['mock_provider']['mock_model'],
          run_type=types.RunType.TEST,
          strict_feature_test=True,
          logging_options=base_logging_options,
          proxdash_connection=proxdash_connection)

      init_state = connector.get_init_state()
      assert init_state.provider_model == model_configs.ALL_MODELS[
          'mock_provider']['mock_model']
      assert init_state.run_type == types.RunType.TEST
      assert init_state.strict_feature_test == True
      assert init_state.logging_options.stdout == True
      assert init_state.logging_options.hide_sensitive_content == True
      assert init_state.logging_options.logging_path == temp_dir
      assert (
          init_state.proxdash_init_state.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert init_state.proxdash_init_state.hidden_run_key == 'test_key'
      assert init_state.proxdash_init_state.api_key == 'test_api_key'
      assert init_state.proxdash_init_state.experiment_path == 'test/path'

  def test_init_with_none_model(self):
    init_state = types.ModelInitState(run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match='provider_model parameter is required'):
      mock_provider.MockProviderModelConnector(init_state=init_state)

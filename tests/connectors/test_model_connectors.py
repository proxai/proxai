import pytest
import datetime
import tempfile
import copy
from typing import Dict, Optional, Callable
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


def get_mock_provider_model_connector(
    feature_mapping_strategy: Optional[types.FeatureMappingStrategy] = None,
    provider_model_config: Optional[types.ProviderModelConfigType] = None,
    query_cache_manager: Optional[query_cache.QueryCacheManager] = None,
    get_query_cache_manager: Optional[
        Callable[[], query_cache.QueryCacheManager]] = None,
    stats: Optional[Dict[str, stats_type.ProviderModelStats]] = None,
):
  model_configs_instance = model_configs.ModelConfigs()
  if provider_model_config is None:
    provider_model_config = model_configs_instance.get_provider_model_config(
        ('mock_provider', 'mock_model'))
  connector = mock_provider.MockProviderModelConnector(
      provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
      run_type=types.RunType.TEST,
      provider_model_config=provider_model_config,
      logging_options=types.LoggingOptions(),
      feature_mapping_strategy=feature_mapping_strategy,
      query_cache_manager=query_cache_manager,
      get_query_cache_manager=get_query_cache_manager,
      stats=stats,
      proxdash_connection=proxdash.ProxDashConnection(
          init_state=types.ProxDashConnectionState(
              status=types.ProxDashConnectionStatus.CONNECTED,
              experiment_path='test/path',
              logging_options=types.LoggingOptions(),
              proxdash_options=types.ProxDashOptions(
                  api_key='test_api_key',
              ),
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

  def test_generic_property(self, model_configs_instance):
    connector = get_mock_provider_model_connector()
    assert connector.provider_model == model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
    assert (
        connector._provider_model_state.provider_model ==
        model_configs_instance.get_provider_model(('mock_provider', 'mock_model')))

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
    assert (
        connector.proxdash_connection.proxdash_options.api_key ==
        'test_api_key')
    assert connector.proxdash_connection.experiment_path == 'test/path'

    assert isinstance(
        connector._provider_model_state.proxdash_connection,
        types.ProxDashConnectionState)
    assert (
        connector._provider_model_state.proxdash_connection ==
        connector.proxdash_connection.get_state())

  def test_proxdash_connection_function(self, model_configs_instance):
    dynamic_proxdash_connection = proxdash.ProxDashConnection(
        init_state=types.ProxDashConnectionState(
            status=types.ProxDashConnectionStatus.CONNECTED,
            experiment_path='test/path',
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(
                api_key='test_api_key',
            ),
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

    def get_proxdash_connection():
      return dynamic_proxdash_connection

    connector = mock_provider.MockProviderModelConnector(
        provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
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
  def test_init_state(self, model_configs_instance):
    init_state = types.ProviderModelState(
        provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
        run_type=types.RunType.TEST,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        logging_options=types.LoggingOptions(stdout=True),
        proxdash_connection=types.ProxDashConnectionState(
            status=types.ProxDashConnectionStatus.CONNECTED,
            hidden_run_key='test_key',
            experiment_path='test/path',
            logging_options=types.LoggingOptions(stdout=True),
            proxdash_options=types.ProxDashOptions(
                api_key='test_api_key',
            ),
            key_info_from_proxdash={'permission': 'ALL'},
            connected_experiment_path='test/path'))

    connector = mock_provider.MockProviderModelConnector(init_state=init_state)

    assert connector.provider_model == model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
    assert connector.run_type == types.RunType.TEST
    assert connector.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
    assert connector.logging_options.stdout == True
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert (
        connector.proxdash_connection.proxdash_options.api_key ==
        'test_api_key')

  def test_init_with_mismatched_model(self, model_configs_instance):
    init_state = types.ProviderModelState(
        provider_model=model_configs_instance.get_provider_model(('claude', 'opus-4')),
        run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match=(
            'provider_model needs to be same with the class provider name.\n'
            'provider_model: *')):
      mock_provider.MockProviderModelConnector(init_state=init_state)

  def test_init_state_with_all_options(self, model_configs_instance):
    with tempfile.TemporaryDirectory() as temp_dir:
      init_state = types.ProviderModelState(
          provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
          logging_options=types.LoggingOptions(
              logging_path=temp_dir,
              hide_sensitive_content=True,
              stdout=True),
          proxdash_connection=types.ProxDashConnectionState(
              status=types.ProxDashConnectionStatus.CONNECTED,
              hidden_run_key='test_key',
              experiment_path='test/path',
              logging_options=types.LoggingOptions(
                  logging_path=temp_dir,
                  hide_sensitive_content=True,
                  stdout=True),
              proxdash_options=types.ProxDashOptions(
                  stdout=True,
                  api_key='test_api_key')))

      connector = mock_provider.MockProviderModelConnector(
          init_state=init_state)

      assert connector.provider_model == model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
      assert connector.run_type == types.RunType.TEST
      assert connector.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
      assert connector.logging_options.stdout == True
      assert connector.logging_options.hide_sensitive_content == True
      assert connector.logging_options.logging_path == temp_dir
      assert (
          connector.proxdash_connection.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert connector.proxdash_connection._hidden_run_key == 'test_key'
      assert (
          connector.proxdash_connection.proxdash_options.api_key ==
          'test_api_key')
      assert connector.proxdash_connection.experiment_path == 'test/path'

  def test_init_with_literals(self, model_configs_instance):
    with tempfile.TemporaryDirectory() as temp_dir:
      base_logging_options = types.LoggingOptions(
          stdout=True,
          hide_sensitive_content=True,
          logging_path=temp_dir)

      proxdash_connection = proxdash.ProxDashConnection(
          hidden_run_key='test_key',
          experiment_path='test/path',
          logging_options=base_logging_options,
          proxdash_options=types.ProxDashOptions(
              stdout=True,
              api_key='test_api_key'))

      connector = mock_provider.MockProviderModelConnector(
          provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
          logging_options=base_logging_options,
          proxdash_connection=proxdash_connection)

      init_state = connector.get_state()
      assert init_state.provider_model == model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
      assert init_state.run_type == types.RunType.TEST
      assert init_state.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
      assert init_state.logging_options.stdout == True
      assert init_state.logging_options.hide_sensitive_content == True
      assert init_state.logging_options.logging_path == temp_dir
      assert (
          init_state.proxdash_connection.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert init_state.proxdash_connection.hidden_run_key == 'test_key'
      assert (
          init_state.proxdash_connection.proxdash_options.api_key ==
          'test_api_key')
      assert init_state.proxdash_connection.experiment_path == 'test/path'

  def test_init_with_none_model(self):
    init_state = types.ProviderModelState(run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match='provider_model needs to be set in init_state.'):
      mock_provider.MockProviderModelConnector(init_state=init_state)

  def test_init_with_invalid_combinations(self, model_configs_instance):
    with pytest.raises(
        ValueError,
        match=(
            'Only one of logging_options or get_logging_options should be set '
            'while initializing the StateControlled object.')):
      mock_provider.MockProviderModelConnector(
          provider_model=model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          logging_options=types.LoggingOptions(stdout=True),
          get_logging_options=lambda: types.LoggingOptions(stdout=True))

  def test_invalid_model_combination(self, model_configs_instance):
    connector = get_mock_provider_model_connector()
    with pytest.raises(
        ValueError,
        match=(
            'provider_model does not match the connector provider_model.'
            'provider_model: *')):
      connector.generate_text(
          prompt="Hello",
          provider_model=model_configs_instance.get_provider_model(('claude', 'opus-4')))


def _get_config_with_not_supported_feature(feature_name: str):
  """Helper to create a config with a not_supported feature."""
  model_configs_instance = model_configs.ModelConfigs()
  base_config = model_configs_instance.get_provider_model_config(
      ('mock_provider', 'mock_model'))
  return types.ProviderModelConfigType(
      provider_model=base_config.provider_model,
      pricing=base_config.pricing,
      features=types.ProviderModelFeatureType(
          not_supported_features=[feature_name]),
      metadata=base_config.metadata)


def _get_config_with_best_effort_feature(feature_name: str):
  """Helper to create a config with a best_effort feature."""
  model_configs_instance = model_configs.ModelConfigs()
  base_config = model_configs_instance.get_provider_model_config(
      ('mock_provider', 'mock_model'))
  return types.ProviderModelConfigType(
      provider_model=base_config.provider_model,
      pricing=base_config.pricing,
      features=types.ProviderModelFeatureType(
          best_effort_features=[feature_name]),
      metadata=base_config.metadata)


class TestFeatureMappingStrategy:
  def test_not_supported_strict_raises(self):
    config = _get_config_with_not_supported_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    with pytest.raises(Exception, match="does not support system"):
      connector.generate_text(prompt="Hello", system="Be helpful")

  def test_not_supported_best_effort_omits(self):
    config = _get_config_with_not_supported_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    result = connector.generate_text(prompt="Hello", system="Be helpful")
    assert result.response_record.response is not None

  def test_not_supported_omit_omits(self):
    config = _get_config_with_not_supported_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.OMIT,
        provider_model_config=config)
    result = connector.generate_text(prompt="Hello", system="Be helpful")
    assert result.response_record.response is not None

  def test_not_supported_passthrough_keeps(self):
    config = _get_config_with_not_supported_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.PASSTHROUGH,
        provider_model_config=config)
    result = connector.generate_text(prompt="Hello", system="Be helpful")
    assert result.response_record.response is not None

  def test_best_effort_strict_raises(self):
    config = _get_config_with_best_effort_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    with pytest.raises(Exception, match="does not support system"):
      connector.generate_text(prompt="Hello", system="Be helpful")

  def test_best_effort_omit_omits(self):
    config = _get_config_with_best_effort_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.OMIT,
        provider_model_config=config)
    result = connector.generate_text(prompt="Hello", system="Be helpful")
    assert result.response_record.response is not None

  def test_best_effort_best_effort_keeps(self):
    config = _get_config_with_best_effort_feature('system')
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    result = connector.generate_text(prompt="Hello", system="Be helpful")
    assert result.response_record.response is not None


class TestModelConnector:
  def test_mutually_exclusive_params(self):
    connector = get_mock_provider_model_connector()
    with pytest.raises(
        ValueError,
        match="prompt and messages cannot be set at the same time"):
      connector.generate_text(
          prompt="Hello",
          messages=[{"role": "user", "content": "Hello"}])

  def test_generate_text(self):
    connector = get_mock_provider_model_connector()
    result = connector.generate_text(
        prompt="Hello",
        max_tokens=100)

    assert isinstance(result, types.LoggingRecord)
    assert result.response_record.response.value == "mock response"
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

  def test_generate_text_with_query_cache_manager_function(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      dynamic_cache_options = types.CacheOptions(cache_path=temp_dir)

      def get_cache_options():
        return dynamic_cache_options

      def get_query_cache_manager():
        return query_cache.QueryCacheManager(
            get_cache_options=get_cache_options)

      connector = get_mock_provider_model_connector(
          get_query_cache_manager=get_query_cache_manager)

      dynamic_cache_options.unique_response_limit = 2
      connector.apply_external_state_changes()

      # First call - should hit the provider
      result1 = connector.generate_text(prompt="Hello")
      assert result1.response_source == types.ResponseSource.PROVIDER

      # Second call - should hit the provider
      result2 = connector.generate_text(prompt="Hello")
      assert result2.response_source == types.ResponseSource.PROVIDER

      # Third call - should hit the cache
      result3 = connector.generate_text(prompt="Hello")
      assert result3.response_source == types.ResponseSource.CACHE

  def test_stats_update(self, model_configs_instance):
    provider_model = model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
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
    assert run_time_stats.provider_stats.total_token_count > 0
    assert run_time_stats.provider_stats.total_query_token_count > 0
    assert run_time_stats.provider_stats.total_response_token_count > 0
    assert run_time_stats.provider_stats.total_response_time > 0
    assert run_time_stats.provider_stats.estimated_cost > 0


class TestGetTokenCountEstimate:
  def test_string_input(self):
    connector = get_mock_provider_model_connector()
    result = connector.get_token_count_estimate('Hello world')
    assert result > 0

  def test_text_response(self):
    connector = get_mock_provider_model_connector()
    response = types.Response(
        type=types.ResponseType.TEXT,
        value='Hello world')
    result = connector.get_token_count_estimate(response)
    assert result > 0

  def test_json_response(self):
    connector = get_mock_provider_model_connector()
    response = types.Response(
        type=types.ResponseType.JSON,
        value={'key': 'value'})
    result = connector.get_token_count_estimate(response)
    assert result > 0

  def test_pydantic_response_with_instance_value(self):
    import pydantic

    class TestModel(pydantic.BaseModel):
      name: str
      value: int

    connector = get_mock_provider_model_connector()
    response = types.Response(
        type=types.ResponseType.PYDANTIC,
        value=types.ResponsePydanticValue(
            instance_value=TestModel(name='test', value=42)))
    result = connector.get_token_count_estimate(response)
    assert result > 0

  def test_pydantic_response_with_instance_json_value(self):
    connector = get_mock_provider_model_connector()
    response = types.Response(
        type=types.ResponseType.PYDANTIC,
        value=types.ResponsePydanticValue(
            instance_json_value={'name': 'test', 'value': 42}))
    result = connector.get_token_count_estimate(response)
    assert result > 0

  def test_messages_input(self):
    connector = get_mock_provider_model_connector()
    messages = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there'}]
    result = connector.get_token_count_estimate(messages)
    assert result > 0

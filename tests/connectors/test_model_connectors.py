import pytest
import json
import datetime
import tempfile
import copy
import functools
from typing import Dict, Optional, Callable
import pydantic
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


def _get_config_with_features(
    not_supported: list = None,
    best_effort: list = None,
    supported: list = None):
  """Helper to create a config with specific feature settings."""
  model_configs_instance = model_configs.ModelConfigs()
  base_config = model_configs_instance.get_provider_model_config(
      ('mock_provider', 'mock_model'))
  return types.ProviderModelConfigType(
      provider_model=base_config.provider_model,
      pricing=base_config.pricing,
      features=types.ProviderModelFeatureType(
          not_supported=not_supported or [],
          best_effort=best_effort or [],
          supported=supported or []),
      metadata=base_config.metadata)


class TestFeatureCheck:
  """Tests for feature_check, handle_feature_not_supported, handle_feature_best_effort.

  Regular features (e.g., system):
  | Feature Type  | STRICT      | BEST_EFFORT |
  |---------------|-------------|-------------|
  | not_supported | raises      | raises      |
  | best_effort   | raises      | omits       |

  response_format:: special syntax:
  | Feature Type  | STRICT      | BEST_EFFORT |
  |---------------|-------------|-------------|
  | not_supported | raises      | raises      |
  | best_effort   | raises      | keeps       |
  """

  # handle_feature_not_supported tests
  def test_not_supported_regular_feature_raises(self):
    config = _get_config_with_features(not_supported=['system'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(system='Be helpful')
    with pytest.raises(Exception, match='does not support system'):
      connector.handle_feature_not_supported(query_record)

  def test_not_supported_feature_not_used_passes(self):
    config = _get_config_with_features(not_supported=['system'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')
    connector.handle_feature_not_supported(query_record)  # Should not raise

  def test_not_supported_response_format_raises(self):
    config = _get_config_with_features(not_supported=['response_format::json'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    with pytest.raises(Exception, match='does not support response_format::json'):
      connector.handle_feature_not_supported(query_record)

  def test_not_supported_response_format_different_type_passes(self):
    config = _get_config_with_features(not_supported=['response_format::json'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.TEXT))
    connector.handle_feature_not_supported(query_record)  # Should not raise

  def test_not_supported_empty_list_passes(self):
    config = _get_config_with_features(not_supported=[])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(system='Be helpful')
    connector.handle_feature_not_supported(query_record)  # Should not raise

  # handle_feature_best_effort tests
  def test_best_effort_strict_raises(self):
    config = _get_config_with_features(best_effort=['system'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        system='Be helpful',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    with pytest.raises(Exception, match='does not support system.*STRICT'):
      connector.handle_feature_best_effort(query_record)

  def test_best_effort_omits_regular_feature(self):
    config = _get_config_with_features(best_effort=['system'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        system='Be helpful',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)
    connector.handle_feature_best_effort(query_record)
    assert query_record.system is None  # Feature was omitted

  def test_best_effort_response_format_strict_raises(self):
    config = _get_config_with_features(best_effort=['response_format::json'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    with pytest.raises(Exception, match='does not support response_format::json.*STRICT'):
      connector.handle_feature_best_effort(query_record)

  def test_best_effort_response_format_keeps(self):
    config = _get_config_with_features(best_effort=['response_format::json'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    connector.handle_feature_best_effort(query_record)
    assert query_record.response_format is not None  # Feature was kept

  def test_best_effort_feature_not_used_passes(self):
    config = _get_config_with_features(best_effort=['system'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')
    connector.handle_feature_best_effort(query_record)  # Should not raise

  # feature_check tests (integration of both handlers)
  def test_feature_check_returns_copy(self):
    config = _get_config_with_features()
    connector = get_mock_provider_model_connector(provider_model_config=config)
    original = types.QueryRecord(system='Be helpful')
    result = connector.feature_check(original)
    assert result is not original  # Returns a copy

  def test_feature_check_not_supported_raises(self):
    config = _get_config_with_features(not_supported=['system'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(system='Be helpful')
    with pytest.raises(Exception, match='does not support system'):
      connector.feature_check(query_record)

  def test_feature_check_best_effort_omits(self):
    config = _get_config_with_features(best_effort=['system'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        system='Be helpful',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)
    result = connector.feature_check(query_record)
    assert result.system is None
    assert query_record.system == 'Be helpful'  # Original unchanged

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


class TestExtractJsonFromText:
  """Tests for _extract_json_from_text helper function."""

  def test_direct_json(self):
    connector = get_mock_provider_model_connector()
    result = connector._extract_json_from_text('{"name": "John", "age": 30}')
    assert result == {"name": "John", "age": 30}

  def test_markdown_code_block_with_json_tag(self):
    connector = get_mock_provider_model_connector()
    text = '```json\n{"name": "John"}\n```'
    assert connector._extract_json_from_text(text) == {"name": "John"}

  def test_markdown_code_block_without_tag(self):
    connector = get_mock_provider_model_connector()
    text = '```\n{"name": "John"}\n```'
    assert connector._extract_json_from_text(text) == {"name": "John"}

  def test_json_with_surrounding_text(self):
    connector = get_mock_provider_model_connector()
    text = 'Here is the result: {"name": "John"} Hope this helps!'
    assert connector._extract_json_from_text(text) == {"name": "John"}

  def test_python_dict_style_single_quotes(self):
    connector = get_mock_provider_model_connector()
    text = "{'name': 'John', 'age': 30}"
    assert connector._extract_json_from_text(text) == {"name": "John", "age": 30}

  def test_whitespace_handling(self):
    connector = get_mock_provider_model_connector()
    text = '  \n  {"name": "John"}  \n  '
    assert connector._extract_json_from_text(text) == {"name": "John"}

  def test_invalid_json_raises(self):
    connector = get_mock_provider_model_connector()
    with pytest.raises(json.JSONDecodeError):
      connector._extract_json_from_text('not json at all')


class SamplePydanticModel(pydantic.BaseModel):
  name: str
  age: int


class TestSystemAndResponseFormatParams:
  """Tests for system message and response format parameter handling.

  Tests cover:
  - _get_system_content_with_schema_guidance
  - _add_response_format_param
  - _add_supported_system_and_response_format_params
  - _add_best_effort_system_and_response_format_params
  - add_system_and_response_format_params
  - format_response_from_providers
  """

  # _get_system_content_with_schema_guidance tests
  def test_schema_guidance_json_without_system(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert result == 'You must respond with valid JSON.'

  def test_schema_guidance_json_with_system(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        system='Be helpful.',
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert result == 'Be helpful.\n\nYou must respond with valid JSON.'

  def test_schema_guidance_json_schema(self):
    connector = get_mock_provider_model_connector()
    schema = {'json_schema': {'schema': {'type': 'object'}}}
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.JSON_SCHEMA,
            value=schema))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert 'You must respond with valid JSON that follows this schema:' in result
    assert '"type": "object"' in result

  def test_schema_guidance_pydantic(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(
                class_value=SamplePydanticModel)))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert 'You must respond with valid JSON that follows this schema:' in result
    assert 'name' in result
    assert 'age' in result

  # add_system_and_response_format_params tests
  def test_add_params_no_response_format(self):
    config = _get_config_with_features(supported=['system'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(system='Be helpful.')
    base_func = lambda: None
    result = connector.add_system_and_response_format_params(base_func, query_record)
    assert result is not None

  def test_add_params_text_response_format(self):
    config = _get_config_with_features(supported=['system'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.TEXT))
    base_func = lambda: None
    result = connector.add_system_and_response_format_params(base_func, query_record)
    assert result is not None

  def test_add_params_supported_json_format(self):
    config = _get_config_with_features(supported=['response_format::json'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    base_func = lambda: None
    result = connector.add_system_and_response_format_params(base_func, query_record)
    assert result is not None

  def test_add_params_best_effort_strict_raises(self):
    config = _get_config_with_features(best_effort=['response_format::json'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        provider_model=connector.provider_model,
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    base_func = lambda: None
    with pytest.raises(Exception, match='does not support.*STRICT'):
      connector.add_system_and_response_format_params(base_func, query_record)

  def test_add_params_best_effort_mode(self):
    config = _get_config_with_features(best_effort=['response_format::json'])
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)
    base_func = lambda: None
    result = connector.add_system_and_response_format_params(base_func, query_record)
    assert result is not None

  def test_add_params_not_supported_raises(self):
    config = _get_config_with_features(not_supported=['response_format::json'])
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        provider_model=connector.provider_model,
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    base_func = lambda: None
    with pytest.raises(Exception, match='does not support response_format::json'):
      connector.add_system_and_response_format_params(base_func, query_record)

  def test_add_params_unknown_feature_raises(self):
    config = _get_config_with_features()  # Empty features
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    base_func = lambda: None
    with pytest.raises(Exception, match='not found in provider model config'):
      connector.add_system_and_response_format_params(base_func, query_record)

  # format_response_from_providers tests
  def test_format_response_text(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord()
    response = connector.format_response_from_providers('hello', query_record)
    assert response.type == types.ResponseType.TEXT

  def test_format_response_json(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    response = connector.format_response_from_providers({'key': 'value'}, query_record)
    assert response.type == types.ResponseType.JSON

  def test_format_response_json_schema(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON_SCHEMA))
    response = connector.format_response_from_providers({'key': 'value'}, query_record)
    assert response.type == types.ResponseType.JSON

  def test_format_response_pydantic_strict(self):
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(class_name='SamplePydanticModel')),
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    mock_instance = SamplePydanticModel(name='John', age=30)
    response = connector.format_response_from_providers(mock_instance, query_record)
    assert response.type == types.ResponseType.PYDANTIC
    assert response.value.class_name == 'SamplePydanticModel'

  def test_format_response_pydantic_best_effort(self):
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(
                class_name='SamplePydanticModel',
                class_value=SamplePydanticModel)),
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)
    response = connector.format_response_from_providers(
        {'name': 'John', 'age': 30}, query_record)
    assert response.type == types.ResponseType.PYDANTIC
    assert response.value.instance_value.name == 'John'
    assert response.value.instance_value.age == 30

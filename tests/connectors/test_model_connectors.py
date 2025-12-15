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


# =============================================================================
# Test Helpers
# =============================================================================

def _create_endpoint_features(feature_endpoint_map: Dict[str, Dict[str, list]]):
  """Create a FeatureMappingType for testing.

  Args:
      feature_endpoint_map: Dict mapping feature names to endpoint config.
          Example: {
              'prompt': {'supported': ['chat'], 'best_effort': []},
              'system': {'supported': ['chat'], 'best_effort': ['completion']},
          }

  Returns:
      Dict[str, EndpointFeatureInfoType] - the features config
  """
  features = {}
  for feature_name, endpoints in feature_endpoint_map.items():
    features[feature_name] = types.EndpointFeatureInfoType(
        supported=endpoints.get('supported', []),
        best_effort=endpoints.get('best_effort', []),
        not_supported=endpoints.get('not_supported', []))
  return features


def _create_config_with_features(feature_endpoint_map: Dict[str, Dict[str, list]]):
  """Create a ProviderModelConfigType with specified feature endpoints.

  Args:
      feature_endpoint_map: Dict mapping feature names to endpoint config.

  Returns:
      ProviderModelConfigType with the specified features.
  """
  model_configs_instance = model_configs.ModelConfigs()
  base_config = model_configs_instance.get_provider_model_config(
      ('mock_provider', 'mock_model'))

  return types.ProviderModelConfigType(
      provider_model=base_config.provider_model,
      pricing=base_config.pricing,
      features=_create_endpoint_features(feature_endpoint_map),
      metadata=base_config.metadata)


def _get_default_mock_features():
  """Get default features for mock_provider that support basic operations."""
  return _create_endpoint_features({
      'prompt': {'supported': ['mock_endpoint'], 'best_effort': []},
      'messages': {'supported': ['mock_endpoint'], 'best_effort': []},
      'system': {'supported': ['mock_endpoint'], 'best_effort': []},
      'max_tokens': {'supported': ['mock_endpoint'], 'best_effort': []},
      'temperature': {'supported': ['mock_endpoint'], 'best_effort': []},
      'stop': {'supported': ['mock_endpoint'], 'best_effort': []},
      'web_search': {'supported': ['mock_endpoint'], 'best_effort': []},
      'response_format::text': {'supported': ['mock_endpoint'], 'best_effort': []},
      'response_format::json': {'supported': ['mock_endpoint'], 'best_effort': []},
      'response_format::json_schema': {'supported': ['mock_endpoint'], 'best_effort': []},
      'response_format::pydantic': {'supported': ['mock_endpoint'], 'best_effort': []},
  })


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
    base_config = model_configs_instance.get_provider_model_config(
        ('mock_provider', 'mock_model'))
    # Mock provider has empty features, so we provide defaults
    provider_model_config = types.ProviderModelConfigType(
        provider_model=base_config.provider_model,
        pricing=base_config.pricing,
        features=_get_default_mock_features(),
        metadata=base_config.metadata)
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


# =============================================================================
# Property and Init Tests
# =============================================================================

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


class TestCheckFeatureExists:
  """Tests for _check_feature_exists method."""

  def test_prompt_exists_when_set(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(prompt='Hello')
    assert connector._check_feature_exists('prompt', query_record) is True

  def test_prompt_not_exists_when_none(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord()
    assert connector._check_feature_exists('prompt', query_record) is False

  def test_system_exists_when_set(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(system='Be helpful')
    assert connector._check_feature_exists('system', query_record) is True

  def test_response_format_json_exists(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    assert connector._check_feature_exists('response_format::json', query_record) is True

  def test_response_format_json_not_exists_for_text(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.TEXT))
    assert connector._check_feature_exists('response_format::json', query_record) is False


class TestGetAvailableEndpoints:
  """Tests for _get_available_endpoints method."""

  def test_single_feature_returns_its_endpoints(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat', 'completion'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')

    supported, best_effort = connector._get_available_endpoints(query_record)

    assert 'chat' in supported
    assert 'completion' in supported

  def test_multiple_features_returns_intersection(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat', 'completion'], 'best_effort': []},
        'system': {'supported': ['chat'], 'best_effort': ['completion']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello', system='Be helpful')

    supported, best_effort = connector._get_available_endpoints(query_record)

    # Only 'chat' is in supported for both features
    assert supported == ['chat']
    # Best effort includes chat (from supported) and completion
    assert 'chat' in best_effort
    assert 'completion' in best_effort

  def test_no_common_supported_endpoints(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': ['completion'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello', system='Be helpful')

    supported, best_effort = connector._get_available_endpoints(query_record)

    assert supported == []
    assert 'chat' in best_effort


class TestCheckEndpointsUsability:
  """Tests for _check_endpoints_usability method."""

  def test_strict_mode_with_supported_endpoints_passes(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)

    # Should not raise
    connector._check_endpoints_usability(['chat'], [], query_record)

  def test_strict_mode_without_supported_endpoints_raises(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)

    with pytest.raises(Exception, match='STRICT mode'):
      connector._check_endpoints_usability([], ['chat'], query_record)

  def test_best_effort_mode_with_best_effort_endpoints_passes(self):
    config = _create_config_with_features({
        'prompt': {'supported': [], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)

    # Should not raise
    connector._check_endpoints_usability([], ['chat'], query_record)

  def test_best_effort_mode_without_any_endpoints_raises(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)

    with pytest.raises(Exception, match='BEST_EFFORT mode'):
      connector._check_endpoints_usability([], [], query_record)


class TestSelectEndpoint:
  """Tests for _select_endpoint method."""

  def test_prefers_supported_over_best_effort(self):
    connector = get_mock_provider_model_connector()
    result = connector._select_endpoint(['supported_ep'], ['best_effort_ep'])
    assert result == 'supported_ep'

  def test_falls_back_to_best_effort(self):
    connector = get_mock_provider_model_connector()
    result = connector._select_endpoint([], ['best_effort_ep'])
    assert result == 'best_effort_ep'

  def test_returns_none_when_no_endpoints(self):
    connector = get_mock_provider_model_connector()
    result = connector._select_endpoint([], [])
    assert result is None


class TestSanitizeSystemFeature:
  """Tests for _sanitize_system_feature method."""

  def test_keeps_system_when_endpoint_supports_it(self):
    config = _create_config_with_features({
        'system': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        system='Be helpful',
        chosen_endpoint='chat')

    result = connector._sanitize_system_feature(query_record)

    assert result.system == 'Be helpful'
    assert result.prompt == 'Hello'

  def test_merges_system_into_prompt_when_not_supported(self):
    config = _create_config_with_features({
        'system': {'supported': ['other'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        system='Be helpful',
        chosen_endpoint='chat')

    result = connector._sanitize_system_feature(query_record)

    assert result.system is None
    assert 'Be helpful' in result.prompt
    assert 'Hello' in result.prompt


class TestSanitizeMessagesFeature:
  """Tests for _sanitize_messages_feature method."""

  def test_keeps_messages_when_endpoint_supports_it(self):
    config = _create_config_with_features({
        'messages': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    messages = [{'role': 'user', 'content': 'Hello'}]
    query_record = types.QueryRecord(
        messages=messages,
        chosen_endpoint='chat')

    result = connector._sanitize_messages_feature(query_record)

    assert result.messages == messages

  def test_converts_messages_to_prompt_when_not_supported(self):
    config = _create_config_with_features({
        'messages': {'supported': ['other'], 'best_effort': ['completion']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    messages = [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there'}
    ]
    query_record = types.QueryRecord(
        messages=messages,
        chosen_endpoint='completion')

    result = connector._sanitize_messages_feature(query_record)

    assert result.messages is None
    assert 'USER: Hello' in result.prompt
    assert 'ASSISTANT: Hi there' in result.prompt


class TestSanitizeWebSearchFeature:
  """Tests for _sanitize_web_search_feature method."""

  def test_keeps_web_search_when_true(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(web_search=True)

    result = connector._sanitize_web_search_feature(query_record)

    assert result.web_search is True

  def test_clears_web_search_when_false(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(web_search=False)

    result = connector._sanitize_web_search_feature(query_record)

    assert result.web_search is None


class TestOmitBestEffortFeature:
  """Tests for _omit_best_effort_feature method."""

  def test_keeps_feature_when_supported(self):
    config = _create_config_with_features({
        'temperature': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        temperature=0.7,
        chosen_endpoint='chat')

    result = connector._omit_best_effort_feature('temperature', query_record)

    assert result.temperature == 0.7

  def test_omits_feature_when_not_supported(self):
    config = _create_config_with_features({
        'temperature': {'supported': ['other'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        temperature=0.7,
        chosen_endpoint='chat')

    result = connector._omit_best_effort_feature('temperature', query_record)

    assert result.temperature is None


class TestGetSchemaGuidance:
  """Tests for _get_schema_guidance method."""

  def test_json_format_guidance(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))

    result = connector._get_schema_guidance(query_record)

    assert 'valid JSON' in result

  def test_json_schema_format_includes_schema(self):
    connector = get_mock_provider_model_connector()
    schema = {'json_schema': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string'}}}}}
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.JSON_SCHEMA,
            value=schema))

    result = connector._get_schema_guidance(query_record)

    assert 'valid JSON' in result
    assert 'schema' in result


class TestSanitizeResponseFormatFeature:
  """Tests for _sanitize_response_format_feature method."""

  def test_adds_schema_guidance_to_prompt(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Get user data',
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))

    result = connector._sanitize_response_format_feature(query_record)

    assert 'Get user data' in result.prompt
    assert 'JSON' in result.prompt

  def test_creates_prompt_if_none(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))

    result = connector._sanitize_response_format_feature(query_record)

    assert result.prompt is not None
    assert 'JSON' in result.prompt


class TestSanitizeQueryRecord:
  """Tests for _sanitize_query_record method."""

  def test_sets_chosen_endpoint(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')

    result = connector._sanitize_query_record(['chat'], [], query_record)

    assert result.chosen_endpoint == 'chat'

  def test_strict_mode_returns_unchanged(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)

    result = connector._sanitize_query_record(['chat'], [], query_record)

    assert result.prompt == 'Hello'
    assert result.chosen_endpoint == 'chat'


# Note: add_features_to_query_function is tested indirectly through generate_text
# Direct testing would require implementing all abstract feature mapping methods


class SamplePydanticModel(pydantic.BaseModel):
  name: str
  age: int


class TestHandleJsonResponseFormat:
  """Tests for _handle_json_response_format method."""

  def test_supported_endpoint_uses_provider_format(self):
    config = _create_config_with_features({
        'response_format::json': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        chosen_endpoint='chat')

    result = connector._handle_json_response_format({'key': 'value'}, query_record)

    assert result.type == types.ResponseType.JSON

  def test_unsupported_endpoint_extracts_json_from_text(self):
    config = _create_config_with_features({
        'response_format::json': {'supported': ['other'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        chosen_endpoint='chat')

    result = connector._handle_json_response_format('{"key": "value"}', query_record)

    assert result.type == types.ResponseType.JSON
    assert result.value == {'key': 'value'}


class TestHandleJsonSchemaResponseFormat:
  """Tests for _handle_json_schema_response_format method."""

  def test_supported_endpoint_uses_provider_format(self):
    config = _create_config_with_features({
        'response_format::json_schema': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON_SCHEMA),
        chosen_endpoint='chat')

    result = connector._handle_json_schema_response_format({'name': 'John'}, query_record)

    assert result.type == types.ResponseType.JSON

  def test_unsupported_endpoint_extracts_json_from_text(self):
    config = _create_config_with_features({
        'response_format::json_schema': {'supported': ['other'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON_SCHEMA),
        chosen_endpoint='chat')

    result = connector._handle_json_schema_response_format('{"name": "John"}', query_record)

    assert result.type == types.ResponseType.JSON
    assert result.value == {'name': 'John'}


class TestHandlePydanticResponseFormat:
  """Tests for _handle_pydantic_response_format method."""

  def test_supported_endpoint_returns_pydantic_instance(self):
    config = _create_config_with_features({
        'response_format::pydantic': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(
                class_name='SamplePydanticModel',
                class_value=SamplePydanticModel)),
        chosen_endpoint='chat')

    mock_instance = SamplePydanticModel(name='John', age=30)
    result = connector._handle_pydantic_response_format(mock_instance, query_record)

    assert result.type == types.ResponseType.PYDANTIC
    assert result.value.class_name == 'SamplePydanticModel'

  def test_unsupported_endpoint_parses_json_to_pydantic(self):
    config = _create_config_with_features({
        'response_format::pydantic': {'supported': ['other'], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(
                class_name='SamplePydanticModel',
                class_value=SamplePydanticModel)),
        chosen_endpoint='chat')

    result = connector._handle_pydantic_response_format('{"name": "John", "age": 30}', query_record)

    assert result.type == types.ResponseType.PYDANTIC
    assert result.value.instance_value.name == 'John'
    assert result.value.instance_value.age == 30


class TestGetEstimatedCost:
  """Tests for get_estimated_cost method."""

  def test_calculates_cost_from_token_counts(self):
    connector = get_mock_provider_model_connector()
    logging_record = types.LoggingRecord(
        query_record=types.QueryRecord(
            prompt='Hello',
            token_count=100),
        response_record=types.QueryResponseRecord(
            response=types.Response(value='Hi', type=types.ResponseType.TEXT),
            token_count=50))

    result = connector.get_estimated_cost(logging_record)

    # mock_provider pricing: query=2.0, response=1.0 per token
    # Expected: 100 * 2.0 + 50 * 1.0 = 250
    assert result == 250

  def test_handles_none_token_counts(self):
    connector = get_mock_provider_model_connector()
    logging_record = types.LoggingRecord(
        query_record=types.QueryRecord(
            prompt='Hello',
            token_count=None),
        response_record=types.QueryResponseRecord(
            response=types.Response(value='Hi', type=types.ResponseType.TEXT),
            token_count=None))

    result = connector.get_estimated_cost(logging_record)

    assert result == 0


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


class TestGetSystemContentWithSchemaGuidance:
  """Tests for _get_system_content_with_schema_guidance method."""

  def test_json_without_system(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert result == 'You must respond with valid JSON.'

  def test_json_with_system(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        system='Be helpful.',
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert result == 'Be helpful.\n\nYou must respond with valid JSON.'

  def test_json_schema_includes_schema(self):
    connector = get_mock_provider_model_connector()
    schema = {'json_schema': {'schema': {'type': 'object'}}}
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.JSON_SCHEMA,
            value=schema))
    result = connector._get_system_content_with_schema_guidance(query_record)
    assert 'You must respond with valid JSON that follows this schema:' in result
    assert '"type": "object"' in result

  def test_pydantic_includes_schema(self):
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


class TestFormatResponseFromProviders:
  """Tests for format_response_from_providers method."""

  def test_text_response(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord()
    response = connector.format_response_from_providers('hello', query_record)
    assert response.type == types.ResponseType.TEXT

  def test_json_response(self):
    config = _create_config_with_features({
        'response_format::json': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON),
        chosen_endpoint='chat')
    response = connector.format_response_from_providers({'key': 'value'}, query_record)
    assert response.type == types.ResponseType.JSON

  def test_pydantic_response(self):
    config = _create_config_with_features({
        'response_format::pydantic': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    query_record = types.QueryRecord(
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(class_name='SamplePydanticModel')),
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        chosen_endpoint='chat')
    mock_instance = SamplePydanticModel(name='John', age=30)
    response = connector.format_response_from_providers(mock_instance, query_record)
    assert response.type == types.ResponseType.PYDANTIC
    assert response.value.class_name == 'SamplePydanticModel'

import json
import tempfile

import pydantic
import pytest

import proxai.caching.query_cache as query_cache
import proxai.connections.proxdash as proxdash
import proxai.connectors.model_configs as model_configs
import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.mock_provider as mock_provider
import proxai.types as types


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


# =============================================================================
# Test Helpers
# =============================================================================

def _create_endpoint_features(feature_endpoint_map: dict[str, dict[str, list]]):
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


def _create_config_with_features(feature_endpoint_map: dict[str, dict[str, list]]):
  """Create a ProviderModelConfigType with specified feature endpoints.

  Args:
      feature_endpoint_map: Dict mapping feature names to endpoint config.

  Returns:
      ProviderModelConfigType with the specified features.
  """
  base_config = pytest.model_configs_instance.get_provider_model_config(
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
    feature_mapping_strategy: types.FeatureMappingStrategy | None = None,
    provider_model_config: types.ProviderModelConfigType | None = None,
    query_cache_manager: query_cache.QueryCacheManager | None = None,
):
  if provider_model_config is None:
    base_config = pytest.model_configs_instance.get_provider_model_config(
        ('mock_provider', 'mock_model'))
    # Mock provider has empty features, so we provide defaults
    provider_model_config = types.ProviderModelConfigType(
        provider_model=base_config.provider_model,
        pricing=base_config.pricing,
        features=_get_default_mock_features(),
        metadata=base_config.metadata)
  mock_provider_model_params = model_connector.ProviderModelConnectorParams(
      provider_model=pytest.model_configs_instance.get_provider_model(
          ('mock_provider', 'mock_model')),
      run_type=types.RunType.TEST,
      provider_model_config=provider_model_config,
      logging_options=types.LoggingOptions(),
      feature_mapping_strategy=feature_mapping_strategy,
      query_cache_manager=query_cache_manager,
      proxdash_connection=proxdash.ProxDashConnection(
          init_from_state=types.ProxDashConnectionState(
              status=types.ProxDashConnectionStatus.CONNECTED,
              experiment_path='test/path',
              logging_options=types.LoggingOptions(),
              proxdash_options=types.ProxDashOptions(
                  api_key='test_api_key',
              ),
              key_info_from_proxdash={'permission': 'ALL'},
              connected_experiment_path='test/path')))
  connector = mock_provider.MockProviderModelConnector(
      init_from_params=mock_provider_model_params)
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

  def test_generic_property(self):
    connector = get_mock_provider_model_connector()
    assert connector.provider_model == pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
    assert (
        connector._provider_model_state.provider_model ==
        pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')))

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


class TestModelConnectorInit:
  def test_init_state(self):
    init_state = types.ProviderModelState(
        provider_model=pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
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

    connector = mock_provider.MockProviderModelConnector(init_from_state=init_state)

    assert connector.provider_model == pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
    assert connector.run_type == types.RunType.TEST
    assert connector.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
    assert connector.logging_options.stdout
    assert (
        connector.proxdash_connection.status ==
        types.ProxDashConnectionStatus.CONNECTED)
    assert (
        connector.proxdash_connection.proxdash_options.api_key ==
        'test_api_key')

  def test_init_with_mismatched_model(self):
    init_state = types.ProviderModelState(
        provider_model=pytest.model_configs_instance.get_provider_model(
            ('claude', 'opus-4')),
        run_type=types.RunType.TEST)

    with pytest.raises(
        ValueError,
        match=(
            'provider_model needs to be same with the class provider name.\n'
            'provider_model: *')):
      mock_provider.MockProviderModelConnector(init_from_state=init_state)

  def test_init_state_with_all_options(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      init_state = types.ProviderModelState(
          provider_model=pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
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
          init_from_state=init_state)

      assert connector.provider_model == pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
      assert connector.run_type == types.RunType.TEST
      assert connector.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
      assert connector.logging_options.stdout
      assert connector.logging_options.hide_sensitive_content
      assert connector.logging_options.logging_path == temp_dir
      assert (
          connector.proxdash_connection.status ==
          types.ProxDashConnectionStatus.CONNECTED)
      assert connector.proxdash_connection._hidden_run_key == 'test_key'
      assert (
          connector.proxdash_connection.proxdash_options.api_key ==
          'test_api_key')
      assert connector.proxdash_connection.experiment_path == 'test/path'

  def test_init_with_literals(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      base_logging_options = types.LoggingOptions(
          stdout=True,
          hide_sensitive_content=True,
          logging_path=temp_dir)

      proxdash_connection_params = proxdash.ProxDashConnectionParams(
          hidden_run_key='test_key',
          experiment_path='test/path',
          logging_options=base_logging_options,
          proxdash_options=types.ProxDashOptions(
              stdout=True,
              api_key='test_api_key'))

      proxdash_connection = proxdash.ProxDashConnection(
          init_from_params=proxdash_connection_params)

      model_connector_params = model_connector.ProviderModelConnectorParams(
          provider_model=pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
          logging_options=base_logging_options,
          proxdash_connection=proxdash_connection)

      connector = mock_provider.MockProviderModelConnector(
          init_from_params=model_connector_params)

      init_state = connector.get_state()
      assert init_state.provider_model == pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model'))
      assert init_state.run_type == types.RunType.TEST
      assert init_state.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
      assert init_state.logging_options.stdout
      assert init_state.logging_options.hide_sensitive_content
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
        match='provider_model needs to be set in init_from_state.'):
      mock_provider.MockProviderModelConnector(init_from_state=init_state)

  def test_init_with_invalid_combinations(self):
    with pytest.raises(
        ValueError,
        match=(
            'init_from_params and init_from_state cannot be set at the same '
            'time.')):
      model_connector_params = model_connector.ProviderModelConnectorParams(
          provider_model=pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          logging_options=types.LoggingOptions(stdout=True))
      model_connector_state = types.ProviderModelState(
          provider_model=pytest.model_configs_instance.get_provider_model(('mock_provider', 'mock_model')),
          run_type=types.RunType.TEST,
          logging_options=types.LoggingOptions(stdout=True))
      mock_provider.MockProviderModelConnector(
          init_from_params=model_connector_params,
          init_from_state=model_connector_state)

  def test_invalid_model_combination(self):
    connector = get_mock_provider_model_connector()
    with pytest.raises(
        ValueError,
        match=(
            'provider_model does not match the connector provider_model.'
            'provider_model: *')):
      connector.generate_text(
          prompt="Hello",
          provider_model=pytest.model_configs_instance.get_provider_model(('claude', 'opus-4')))


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


class TestGetFeaturesFromQueryRecord:
  """Tests for _get_features_from_query_record method."""

  def test_prompt_only(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(prompt='Hello')
    features = connector._get_features_from_query_record(query_record)
    assert types.FeatureNameType.PROMPT in features
    assert len(features) == 1

  def test_multiple_features(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        system='Be helpful',
        max_tokens=100)
    features = connector._get_features_from_query_record(query_record)
    assert types.FeatureNameType.PROMPT in features
    assert types.FeatureNameType.SYSTEM in features
    assert types.FeatureNameType.MAX_TOKENS in features
    assert len(features) == 3

  def test_response_format_json(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    features = connector._get_features_from_query_record(query_record)
    assert types.FeatureNameType.PROMPT in features
    assert types.FeatureNameType.RESPONSE_FORMAT_JSON in features

  def test_response_format_pydantic(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        response_format=types.ResponseFormat(
            type=types.ResponseFormatType.PYDANTIC,
            value=types.ResponseFormatPydanticValue(
                class_name='TestModel',
                class_value=SamplePydanticModel)))
    features = connector._get_features_from_query_record(query_record)
    assert types.FeatureNameType.RESPONSE_FORMAT_PYDANTIC in features

  def test_empty_query_record(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord()
    features = connector._get_features_from_query_record(query_record)
    assert len(features) == 0


class TestGetAvailableEndpoints:
  """Tests for _get_available_endpoints method."""

  def test_single_feature_returns_its_endpoints(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat', 'completion'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')

    features = connector._get_features_from_query_record(query_record)
    supported, best_effort = connector._get_available_endpoints(features=features)

    assert 'chat' in supported
    assert 'completion' in supported

  def test_multiple_features_returns_intersection(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat', 'completion'], 'best_effort': []},
        'system': {'supported': ['chat'], 'best_effort': ['completion']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello', system='Be helpful')

    features = connector._get_features_from_query_record(query_record)
    supported, best_effort = connector._get_available_endpoints(features=features)

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

    features = connector._get_features_from_query_record(query_record)
    supported, best_effort = connector._get_available_endpoints(features=features)

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
    connector._check_endpoints_usability(
        supported_endpoints=['chat'],
        best_effort_endpoints=[],
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=connector._get_features_from_query_record(query_record))

  def test_strict_mode_without_supported_endpoints_raises(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)

    with pytest.raises(Exception, match='STRICT mode'):
      connector._check_endpoints_usability(
          supported_endpoints=[],
          best_effort_endpoints=['chat'],
          provider_model=query_record.provider_model,
          feature_mapping_strategy=query_record.feature_mapping_strategy,
          features=connector._get_features_from_query_record(query_record))

  def test_best_effort_mode_with_best_effort_endpoints_passes(self):
    config = _create_config_with_features({
        'prompt': {'supported': [], 'best_effort': ['chat']}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)

    # Should not raise
    connector._check_endpoints_usability(
        supported_endpoints=[],
        best_effort_endpoints=['chat'],
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=connector._get_features_from_query_record(query_record))

  def test_best_effort_mode_without_any_endpoints_raises(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT)

    with pytest.raises(Exception, match='BEST_EFFORT mode'):
      connector._check_endpoints_usability(
          supported_endpoints=[],
          best_effort_endpoints=[],
          provider_model=query_record.provider_model,
          feature_mapping_strategy=query_record.feature_mapping_strategy,
          features=connector._get_features_from_query_record(query_record))


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


class TestGetFeatureCheckResultEndpoint:
  """Tests for _get_feature_check_result_endpoint method."""

  def test_returns_supported_endpoint(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    result = connector._get_feature_check_result_endpoint(
        provider_model=connector.provider_model,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        features=[types.FeatureNameType.PROMPT])
    assert result == 'chat'

  def test_returns_best_effort_endpoint(self):
    config = _create_config_with_features({
        'prompt': {'supported': [], 'best_effort': ['completion']}
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    result = connector._get_feature_check_result_endpoint(
        provider_model=connector.provider_model,
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        features=[types.FeatureNameType.PROMPT])
    assert result == 'completion'

  def test_caches_result(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    assert len(connector._chosen_endpoint_cached_result) == 0

    connector._get_feature_check_result_endpoint(
        provider_model=connector.provider_model,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        features=[types.FeatureNameType.PROMPT])

    assert len(connector._chosen_endpoint_cached_result) == 1

  def test_uses_cached_result(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)

    # First call
    connector._get_feature_check_result_endpoint(
        provider_model=connector.provider_model,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        features=[types.FeatureNameType.PROMPT])

    # Modify cache to verify it's being used
    signature = list(connector._chosen_endpoint_cached_result.keys())[0]
    connector._chosen_endpoint_cached_result[signature] = 'cached_endpoint'

    # Second call should return cached value
    result2 = connector._get_feature_check_result_endpoint(
        provider_model=connector.provider_model,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        features=[types.FeatureNameType.PROMPT])

    assert result2 == 'cached_endpoint'


class TestCheckFeatureCompatibility:
  """Tests for check_feature_compatibility method."""

  def test_compatible_single_feature(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    assert connector.check_feature_compatibility([types.FeatureNameType.PROMPT]) is True

  def test_compatible_multiple_features(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': ['chat'], 'best_effort': []},
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    result = connector.check_feature_compatibility([
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.SYSTEM])
    assert result is True

  def test_incompatible_features_strict_mode(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': ['completion'], 'best_effort': []},
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        provider_model_config=config)
    assert connector.check_feature_compatibility([
          types.FeatureNameType.PROMPT,
          types.FeatureNameType.SYSTEM]) is False

  def test_best_effort_mode_fallback(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': [], 'best_effort': ['chat']},
    })
    connector = get_mock_provider_model_connector(
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
        provider_model_config=config)
    result = connector.check_feature_compatibility([
        types.FeatureNameType.PROMPT,
        types.FeatureNameType.SYSTEM])
    assert result is True

  def test_empty_features_list(self):
    connector = get_mock_provider_model_connector()
    result = connector.check_feature_compatibility([])
    # Empty features returns False (no endpoint found)
    assert result is False


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

  def test_sanitizes_with_chosen_endpoint(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': ['chat'], 'best_effort': []},
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        chosen_endpoint='chat')

    result = connector._sanitize_query_record(query_record)

    assert result.chosen_endpoint == 'chat'
    assert result.prompt == 'Hello'

  def test_strict_mode_returns_unchanged(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(
        prompt='Hello',
        chosen_endpoint='chat',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)

    result = connector._sanitize_query_record(query_record)

    assert result.prompt == 'Hello'
    assert result.chosen_endpoint == 'chat'


class TestGetFeatureSignature:
  """Tests for _get_feature_signature method."""

  def test_basic_prompt_signature(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(prompt='Hello')
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    assert 'prompt' in result
    assert 'None' in result  # provider_model and feature_mapping_strategy are not included in the signature

  def test_signature_includes_provider_model(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        provider_model=pytest.model_configs_instance.get_provider_model(
            ('mock_provider', 'mock_model')))
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    assert 'mock_provider' in result
    assert 'mock_model' in result

  def test_signature_includes_feature_mapping_strategy(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    assert 'STRICT' in result

  def test_signature_with_response_format_json(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        response_format=types.ResponseFormat(type=types.ResponseFormatType.JSON))
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    assert 'response_format::json' in result
    assert 'prompt' in result

  def test_signature_without_response_format(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(prompt='Hello')
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    # Should not raise error and should not include response_format
    assert 'response_format::json' not in result
    assert 'prompt' in result

  def test_signature_with_multiple_features(self):
    connector = get_mock_provider_model_connector()
    query_record = types.QueryRecord(
        prompt='Hello',
        system='Be helpful',
        max_tokens=100,
        temperature=0.7)
    features = connector._get_features_from_query_record(query_record)
    result = connector._get_feature_signature(
        provider_model=query_record.provider_model,
        feature_mapping_strategy=query_record.feature_mapping_strategy,
        features=features)

    assert 'prompt' in result
    assert 'system' in result
    assert 'max_tokens' in result
    assert 'temperature' in result

  def test_same_features_produce_same_signature(self):
    connector = get_mock_provider_model_connector()
    query_record1 = types.QueryRecord(
        prompt='Hello world',
        system='Be helpful')
    query_record2 = types.QueryRecord(
        prompt='Different prompt',
        system='Different system')
    features1 = connector._get_features_from_query_record(query_record1)
    features2 = connector._get_features_from_query_record(query_record2)
    result1 = connector._get_feature_signature(
        provider_model=query_record1.provider_model,
        feature_mapping_strategy=query_record1.feature_mapping_strategy,
        features=features1)
    result2 = connector._get_feature_signature(
        provider_model=query_record2.provider_model,
        feature_mapping_strategy=query_record2.feature_mapping_strategy,
        features=features2)

    # Same features used, so same signature (content doesn't matter)
    assert result1 == result2

  def test_different_features_produce_different_signatures(self):
    connector = get_mock_provider_model_connector()
    query_record1 = types.QueryRecord(prompt='Hello')
    query_record2 = types.QueryRecord(prompt='Hello', system='Be helpful')
    features1 = connector._get_features_from_query_record(query_record1)
    features2 = connector._get_features_from_query_record(query_record2)
    result1 = connector._get_feature_signature(
        provider_model=query_record1.provider_model,
        feature_mapping_strategy=query_record1.feature_mapping_strategy,
        features=features1)
    result2 = connector._get_feature_signature(
        provider_model=query_record2.provider_model,
        feature_mapping_strategy=query_record2.feature_mapping_strategy,
        features=features2)

    assert result1 != result2


class TestChosenEndpointCachedResult:
  """Tests for _chosen_endpoint_cached_result caching logic."""

  def test_cache_is_initialized_empty(self):
    connector = get_mock_provider_model_connector()
    assert connector._chosen_endpoint_cached_result == {}

  def test_feature_check_caches_endpoint_result(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')

    # First call should populate the cache
    result1 = connector.feature_check_and_sanitize(query_record)

    # Cache should now contain the result
    assert len(connector._chosen_endpoint_cached_result) == 1
    assert result1.chosen_endpoint == 'chat'

  def test_feature_check_uses_cached_result(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record = types.QueryRecord(prompt='Hello')

    # First call
    result1 = connector.feature_check_and_sanitize(query_record)
    cache_size_after_first = len(connector._chosen_endpoint_cached_result)

    # Second call with same features
    result2 = connector.feature_check_and_sanitize(query_record)
    cache_size_after_second = len(connector._chosen_endpoint_cached_result)

    # Cache size should not increase
    assert cache_size_after_first == cache_size_after_second
    assert result1.chosen_endpoint == result2.chosen_endpoint

  def test_different_features_create_different_cache_entries(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []},
        'system': {'supported': ['chat'], 'best_effort': []},
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record1 = types.QueryRecord(prompt='Hello')
    query_record2 = types.QueryRecord(prompt='Hello', system='Be helpful')

    connector.feature_check_and_sanitize(query_record1)
    connector.feature_check_and_sanitize(query_record2)

    # Two different feature combinations should create two cache entries
    assert len(connector._chosen_endpoint_cached_result) == 2

  def test_same_features_different_content_uses_cache(self):
    config = _create_config_with_features({
        'prompt': {'supported': ['chat'], 'best_effort': []}
    })
    connector = get_mock_provider_model_connector(provider_model_config=config)
    query_record1 = types.QueryRecord(prompt='Hello')
    query_record2 = types.QueryRecord(prompt='Goodbye')

    connector.feature_check_and_sanitize(query_record1)
    connector.feature_check_and_sanitize(query_record2)

    # Same features (just prompt), so only one cache entry
    assert len(connector._chosen_endpoint_cached_result) == 1


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
    assert result.pydantic_metadata.class_name == 'SamplePydanticModel'

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
    assert result.value.name == 'John'
    assert result.value.age == 30


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

    # mock_provider pricing: query=1.0, response=2.0 per token
    # Expected: 100 * 1.0 + 50 * 2.0 = 200
    assert result == 200

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
      query_cache_params = query_cache.QueryCacheManagerParams(
          cache_options=types.CacheOptions(cache_path=temp_dir))
      cache_manager = query_cache.QueryCacheManager(
          init_from_params=query_cache_params)

      connector = get_mock_provider_model_connector(
          query_cache_manager=cache_manager)

      # First call - should hit the provider
      result1 = connector.generate_text(prompt="Hello")
      assert result1.response_source == types.ResponseSource.PROVIDER

      # Second call - should hit the cache
      result2 = connector.generate_text(prompt="Hello")
      assert result2.response_source == types.ResponseSource.CACHE


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
        value=TestModel(name='test', value=42),
        pydantic_metadata=types.PydanticMetadataType(
            class_name='TestModel'))
    result = connector.get_token_count_estimate(response)
    assert result > 0

  def test_pydantic_response_with_instance_json_value(self):
    connector = get_mock_provider_model_connector()
    response = types.Response(
        type=types.ResponseType.PYDANTIC,
        value=None,
        pydantic_metadata=types.PydanticMetadataType(
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
    assert response.pydantic_metadata.class_name == 'SamplePydanticModel'

import os
import tempfile
import pytest
import proxai.types as types
import proxai.client as client
import proxai.connectors.model_configs as model_configs


@pytest.fixture(autouse=True)
def setup_test(monkeypatch):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.setenv(api_key, 'test_api_key')
  yield


def _get_path_dir(temp_path: str):
  temp_dir = tempfile.TemporaryDirectory()
  path = os.path.join(temp_dir.name, temp_path)
  os.makedirs(path, exist_ok=True)
  return path, temp_dir


def _create_default_params(**kwargs):
  """Helper to create ProxAIClientParams with defaults."""
  return client.ProxAIClientParams(**kwargs)


def _create_client_from_params(**kwargs):
  """Helper to create ProxAIClient from params."""
  params = _create_default_params(**kwargs)
  return client.ProxAIClient(init_from_params=params)


def _create_test_client(**kwargs):
  """Helper to create ProxAIClient for testing with mock provider."""
  px_client = _create_client_from_params(**kwargs)
  px_client._available_models_instance.run_type = types.RunType.TEST
  px_client.set_model(('mock_provider', 'mock_model'))
  return px_client


class TestProxAIClientParams:
  """Test ProxAIClientParams dataclass defaults and structure."""

  def test_default_values(self):
    params = client.ProxAIClientParams()
    assert params.experiment_path is None
    assert params.cache_options is None
    assert params.logging_options is None
    assert params.proxdash_options is None
    assert params.allow_multiprocessing == True
    assert params.model_test_timeout == 25
    assert params.feature_mapping_strategy == types.FeatureMappingStrategy.BEST_EFFORT
    assert params.suppress_provider_errors == False

  def test_custom_values(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    params = client.ProxAIClientParams(
        experiment_path='test/experiment',
        cache_options=cache_options,
        allow_multiprocessing=False,
        model_test_timeout=30,
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        suppress_provider_errors=True)
    assert params.experiment_path == 'test/experiment'
    assert params.cache_options == cache_options
    assert params.allow_multiprocessing == False
    assert params.model_test_timeout == 30
    assert params.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT
    assert params.suppress_provider_errors == True


class TestProxAIClientInit:
  """Test ProxAIClient initialization."""

  def test_init_from_default_params(self):
    px_client = _create_client_from_params()
    assert px_client.run_type == types.RunType.PRODUCTION
    assert px_client.allow_multiprocessing == True
    assert px_client.model_test_timeout == 25
    assert px_client.feature_mapping_strategy == types.FeatureMappingStrategy.BEST_EFFORT
    assert px_client.suppress_provider_errors == False

  def test_init_from_params_with_cache_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(cache_options=cache_options)
    assert px_client.cache_options.cache_path == cache_path
    assert px_client.model_cache_manager is not None
    assert px_client.query_cache_manager is not None

  def test_init_from_params_with_logging_options(self):
    logging_path, _ = _get_path_dir('test_logging')
    logging_options = types.LoggingOptions(logging_path=logging_path)
    px_client = _create_client_from_params(logging_options=logging_options)
    assert px_client.logging_options.logging_path == logging_path

  def test_init_from_params_with_proxdash_disabled(self):
    proxdash_options = types.ProxDashOptions(disable_proxdash=True)
    px_client = _create_client_from_params(proxdash_options=proxdash_options)
    assert px_client.proxdash_options.disable_proxdash == True
    assert (
        px_client.proxdash_connection.status ==
        types.ProxDashConnectionStatus.DISABLED)

  def test_init_from_params_with_experiment_path(self):
    logging_path, _ = _get_path_dir('test_logging')
    logging_options = types.LoggingOptions(logging_path=logging_path)
    px_client = _create_client_from_params(
        experiment_path='my/experiment',
        logging_options=logging_options)
    assert px_client.experiment_path == 'my/experiment'
    assert px_client.logging_options.logging_path == os.path.join(
        logging_path, 'my/experiment')

  def test_init_conflicting_args_raises_error(self):
    with pytest.raises(
        ValueError,
        match='init_from_params and init_from_state cannot be set at the same time.'):
      client.ProxAIClient(
          init_from_params=client.ProxAIClientParams(),
          init_from_state=types.ProxAIClientState())

  def test_clear_model_cache_on_connect(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(
        cache_path=cache_path,
        clear_model_cache_on_connect=True)
    px_client = _create_client_from_params(cache_options=cache_options)
    assert px_client.cache_options.clear_model_cache_on_connect == True

  def test_clear_query_cache_on_connect(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(
        cache_path=cache_path,
        clear_query_cache_on_connect=True)
    px_client = _create_client_from_params(cache_options=cache_options)
    assert px_client.cache_options.clear_query_cache_on_connect == True


class TestProxAIClientPropertyGettersSetters:
  """Test ProxAIClient property getters and setters."""

  def test_run_type(self):
    px_client = _create_client_from_params()
    assert px_client.run_type == types.RunType.PRODUCTION
    px_client.run_type = types.RunType.TEST
    assert px_client.run_type == types.RunType.TEST

  def test_hidden_run_key(self):
    px_client = _create_client_from_params()
    assert px_client.hidden_run_key is not None
    px_client.hidden_run_key = 'custom_key'
    assert px_client.hidden_run_key == 'custom_key'

  def test_experiment_path_valid(self):
    px_client = _create_client_from_params()
    px_client.experiment_path = 'valid/path'
    assert px_client.experiment_path == 'valid/path'

  def test_experiment_path_invalid_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(ValueError):
      px_client.experiment_path = '////invalid_path'

  def test_root_logging_path_valid(self):
    logging_path, _ = _get_path_dir('test_logging')
    px_client = _create_client_from_params()
    px_client.root_logging_path = logging_path
    assert px_client.root_logging_path == logging_path

  def test_root_logging_path_nonexistent_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(ValueError, match='Root logging path does not exist'):
      px_client.root_logging_path = '/nonexistent/path'

  def test_logging_options_with_path(self):
    logging_path, _ = _get_path_dir('test_logging')
    px_client = _create_client_from_params()
    px_client.logging_options = types.LoggingOptions(logging_path=logging_path)
    assert px_client.logging_options.logging_path == logging_path

  def test_logging_options_with_experiment_path(self):
    logging_path, _ = _get_path_dir('test_logging')
    logging_options = types.LoggingOptions(logging_path=logging_path)
    px_client = _create_client_from_params(
        experiment_path='test/exp',
        logging_options=logging_options)
    assert px_client.logging_options.logging_path == os.path.join(
        logging_path, 'test/exp')

  def test_logging_options_creates_directory_if_not_exists(self):
    temp_dir = tempfile.TemporaryDirectory()
    new_logging_path = os.path.join(temp_dir.name, 'new_subdir')
    assert not os.path.exists(new_logging_path)
    logging_options = types.LoggingOptions(logging_path=new_logging_path)
    _create_client_from_params(logging_options=logging_options)
    assert os.path.exists(new_logging_path)

  def test_cache_options_requires_path(self):
    px_client = _create_client_from_params()
    with pytest.raises(ValueError, match='cache_path is required'):
      px_client.cache_options = types.CacheOptions()

  def test_cache_options_with_disable_model_cache(self):
    px_client = _create_client_from_params()
    px_client.cache_options = types.CacheOptions(disable_model_cache=True)
    assert px_client.cache_options.disable_model_cache == True

  def test_proxdash_options(self):
    px_client = _create_client_from_params()
    px_client.proxdash_options = types.ProxDashOptions(stdout=True)
    assert px_client.proxdash_options.stdout == True

  def test_allow_multiprocessing(self):
    px_client = _create_client_from_params()
    assert px_client.allow_multiprocessing == True
    px_client.allow_multiprocessing = False
    assert px_client.allow_multiprocessing == False

  def test_model_test_timeout_valid(self):
    px_client = _create_client_from_params()
    px_client.model_test_timeout = 30
    assert px_client.model_test_timeout == 30

  def test_model_test_timeout_invalid_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(ValueError, match='model_test_timeout must be greater than 0'):
      px_client.model_test_timeout = 0

  def test_feature_mapping_strategy(self):
    px_client = _create_client_from_params()
    assert px_client.feature_mapping_strategy == types.FeatureMappingStrategy.BEST_EFFORT
    px_client.feature_mapping_strategy = types.FeatureMappingStrategy.STRICT
    assert px_client.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT

  def test_suppress_provider_errors(self):
    px_client = _create_client_from_params()
    assert px_client.suppress_provider_errors == False
    px_client.suppress_provider_errors = True
    assert px_client.suppress_provider_errors == True


class TestProxAIClientCacheManagers:
  """Test ProxAIClient cache manager initialization and access."""

  def test_default_model_cache_manager_created(self):
    px_client = _create_client_from_params()
    assert px_client.default_model_cache_manager is not None
    assert px_client.default_model_cache_path is not None

  def test_model_cache_manager_uses_default_when_no_options(self):
    px_client = _create_client_from_params()
    assert px_client.model_cache_manager == px_client.default_model_cache_manager

  def test_model_cache_manager_with_cache_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(cache_options=cache_options)
    assert px_client.model_cache_manager is not None
    assert px_client.model_cache_manager != px_client.default_model_cache_manager

  def test_query_cache_manager_none_without_cache_options(self):
    px_client = _create_client_from_params()
    assert px_client.query_cache_manager is None

  def test_query_cache_manager_with_cache_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(cache_options=cache_options)
    assert px_client.query_cache_manager is not None
    assert (
        px_client.query_cache_manager.status ==
        types.QueryCacheManagerStatus.WORKING)


class TestProxAIClientSetModel:
  """Test ProxAIClient.set_model functionality."""

  def test_set_model_with_provider_model(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(provider_model=('mock_provider', 'mock_model'))
    connector = px_client.registered_model_connectors[types.CallType.GENERATE_TEXT]
    assert connector.provider_model.provider == 'mock_provider'
    assert connector.provider_model.model == 'mock_model'

  def test_set_model_with_generate_text(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(generate_text=('mock_provider', 'mock_model'))
    connector = px_client.registered_model_connectors[types.CallType.GENERATE_TEXT]
    assert connector.provider_model.provider == 'mock_provider'
    assert connector.provider_model.model == 'mock_model'

  def test_set_model_with_tuple(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(('mock_provider', 'mock_model'))
    connector = px_client.registered_model_connectors[types.CallType.GENERATE_TEXT]
    assert connector.provider_model.provider == 'mock_provider'
    assert connector.provider_model.model == 'mock_model'

  def test_set_model_both_params_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(
        ValueError,
        match='provider_model and generate_text cannot be set at the same time'):
      px_client.set_model(
          provider_model=('openai', 'gpt-4'),
          generate_text=('claude', 'haiku-4.5'))

  def test_set_model_no_params_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(
        ValueError,
        match='provider_model or generate_text must be set'):
      px_client.set_model()

  def test_set_model_overwrites_previous_model(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(('mock_provider', 'mock_model'))
    px_client.set_model(('mock_failing_provider', 'mock_failing_model'))
    connector = px_client.registered_model_connectors[types.CallType.GENERATE_TEXT]
    assert connector.provider_model.provider == 'mock_failing_provider'
    assert connector.provider_model.model == 'mock_failing_model'


class TestProxAIClientGenerateText:
  """Test ProxAIClient.generate_text functionality."""

  def test_generate_text_with_prompt(self):
    px_client = _create_test_client()
    response = px_client.generate_text(prompt='hello')
    assert response == 'mock response'

  def test_generate_text_with_messages(self):
    px_client = _create_test_client()
    response = px_client.generate_text(
        messages=[{'role': 'user', 'content': 'hello'}])
    assert response == 'mock response'

  def test_generate_text_prompt_and_messages_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(
        ValueError,
        match='prompt and messages cannot be set at the same time'):
      px_client.generate_text(
          prompt='hello',
          messages=[{'role': 'user', 'content': 'hello'}])

  def test_generate_text_with_provider_model(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    response = px_client.generate_text(
        prompt='hello',
        provider_model=('mock_provider', 'mock_model'))
    assert response == 'mock response'

  def test_generate_text_with_extensive_return(self):
    px_client = _create_test_client()
    response = px_client.generate_text(
        prompt='hello',
        extensive_return=True)
    assert isinstance(response, types.LoggingRecord)
    assert response.response_record.response.value == 'mock response'

  def test_generate_text_with_suppress_provider_errors(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    response = px_client.generate_text(
        prompt='hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True)
    assert response == 'Temp Error'

  def test_generate_text_error_raises_exception_without_suppress(self):
    px_client = _create_client_from_params()
    px_client._available_models_instance.run_type = types.RunType.TEST
    with pytest.raises(Exception, match='Temp Error'):
      px_client.generate_text(
          prompt='hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'))

  def test_generate_text_uses_client_level_suppress_provider_errors(self):
    px_client = _create_client_from_params(suppress_provider_errors=True)
    px_client._available_models_instance.run_type = types.RunType.TEST
    response = px_client.generate_text(
        prompt='hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'))
    assert response == 'Temp Error'

  def test_generate_text_use_cache_without_cache_manager_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(
        ValueError,
        match='use_cache is True but query cache is not working'):
      px_client.generate_text(prompt='hello', use_cache=True)

  def test_generate_text_auto_enables_cache_when_available(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(cache_options=cache_options)
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(('mock_provider', 'mock_model'))

    # First call - should be from provider
    response1 = px_client.generate_text(prompt='hello', extensive_return=True)
    assert response1.response_source == types.ResponseSource.PROVIDER

    # Second call - should be from cache
    response2 = px_client.generate_text(prompt='hello', extensive_return=True)
    assert response2.response_source == types.ResponseSource.CACHE

  def test_generate_text_with_use_cache_false_skips_cache(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(cache_options=cache_options)
    px_client._available_models_instance.run_type = types.RunType.TEST
    px_client.set_model(('mock_provider', 'mock_model'))

    # First call with cache disabled
    response1 = px_client.generate_text(
        prompt='hello', use_cache=False, extensive_return=True)
    assert response1.response_source == types.ResponseSource.PROVIDER

    # Second call with cache disabled - still from provider
    response2 = px_client.generate_text(
        prompt='hello', use_cache=False, extensive_return=True)
    assert response2.response_source == types.ResponseSource.PROVIDER

  def test_generate_text_with_system_prompt(self):
    px_client = _create_test_client()
    response = px_client.generate_text(
        prompt='hello',
        system='You are a helpful assistant.')
    assert response == 'mock response'


class TestProxAIClientGetCurrentOptions:
  """Test ProxAIClient.get_current_options functionality."""

  def test_get_current_options_returns_run_options(self):
    cache_path, _ = _get_path_dir('test_cache')
    cache_options = types.CacheOptions(cache_path=cache_path)
    px_client = _create_client_from_params(
        cache_options=cache_options,
        allow_multiprocessing=False,
        model_test_timeout=30)

    options = px_client.get_current_options()

    assert isinstance(options, types.RunOptions)
    assert options.run_type == types.RunType.PRODUCTION
    assert options.allow_multiprocessing == False
    assert options.model_test_timeout == 30
    assert options.cache_options.cache_path == cache_path

  def test_get_current_options_json_format(self):
    px_client = _create_client_from_params()
    options = px_client.get_current_options(json=True)

    assert isinstance(options, dict)
    assert options['run_type'] == types.RunType.PRODUCTION.value
    assert options['allow_multiprocessing'] == True
    assert options['model_test_timeout'] == 25


class TestProxAIClientGetRegisteredModelConnector:
  """Test ProxAIClient.get_registered_model_connector functionality."""

  def test_get_registered_model_connector_returns_set_model(self):
    px_client = _create_test_client()
    connector = px_client.get_registered_model_connector(types.CallType.GENERATE_TEXT)
    assert connector is not None
    assert connector.provider_model.provider == 'mock_provider'
    assert connector.provider_model.model == 'mock_model'

  def test_get_registered_model_connector_unsupported_call_type_raises_error(self):
    px_client = _create_client_from_params()
    with pytest.raises(ValueError, match='Call type not supported'):
      px_client.get_registered_model_connector('unsupported_call_type')


class TestProxAIClientState:
  """Test ProxAIClient state serialization."""

  def test_get_state_returns_client_state(self):
    px_client = _create_client_from_params(
        experiment_path='test/exp',
        allow_multiprocessing=False,
        model_test_timeout=30)
    state = px_client.get_state()
    assert isinstance(state, types.ProxAIClientState)
    assert state.experiment_path == 'test/exp'
    assert state.allow_multiprocessing == False
    assert state.model_test_timeout == 30

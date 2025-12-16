import os
import platformdirs
import proxai.types as types
from proxai import proxai
import proxai.connectors.model_configs as model_configs
import proxai.caching.model_cache as model_cache
import pytest
import tempfile
import requests


@pytest.fixture(autouse=True)
def setup_test(monkeypatch):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.setenv(api_key, 'test_api_key')
  proxai.reset_state()
  yield


class TestRunType:
    def test_setup_run_type(self):
      proxai.set_run_type(types.RunType.TEST)
      assert proxai._RUN_TYPE == types.RunType.TEST


class TestInitExperimentPath:
  def test_valid_path(self):
    assert proxai._set_experiment_path('test_experiment') == 'test_experiment'

  def test_empty_string(self):
    assert proxai._set_experiment_path() is None

  def test_invalid_path(self):
    with pytest.raises(ValueError):
      proxai._set_experiment_path('////invalid_path')

  def test_invalid_type(self):
    with pytest.raises(TypeError):
      proxai._set_experiment_path(123)

  def test_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_experiment_path('test_experiment', global_set=True)
    assert proxai._EXPERIMENT_PATH == 'test_experiment'

  def test_global_init_none(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_experiment_path(None, global_set=True)
    assert proxai._EXPERIMENT_PATH is None

  def test_global_init_multiple(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_experiment_path('test_1', global_set=True)
    proxai._set_experiment_path('test_2', global_set=True)
    assert proxai._EXPERIMENT_PATH == 'test_2'


class TestInitLoggingPath:

  def test_valid_path(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options, root_logging_path = proxai._set_logging_options(
          logging_path=logging_path)
      assert logging_options.logging_path == logging_path
      assert root_logging_path == logging_path

  def test_experiment_path(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options, root_logging_path = proxai._set_logging_options(
          logging_path=logging_path, experiment_path='test_experiment')
      assert logging_options.logging_path == os.path.join(
          logging_path, 'test_experiment')
      assert root_logging_path == logging_path

  def test_logging_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      logging_options = types.LoggingOptions(logging_path=logging_path)
      logging_options, root_logging_path = proxai._set_logging_options(
          logging_options=logging_options)
      assert logging_options.logging_path == logging_path
      assert root_logging_path == logging_path

  def test_both_logging_path_and_logging_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      with pytest.raises(ValueError):
        proxai._set_logging_options(
            logging_path=logging_path,
            logging_options=types.LoggingOptions(logging_path=logging_path))

  def test_not_exist_logging_path(self):
    with pytest.raises(ValueError):
      proxai._set_logging_options(logging_path='not_exist_logging_path')

  def test_not_exist_logging_options(self):
    with pytest.raises(ValueError):
      logging_options = types.LoggingOptions(
          logging_path='not_exist_logging_path')
      proxai._set_logging_options(logging_options=logging_options)

  def test_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    with tempfile.TemporaryDirectory() as logging_path:
      proxai._set_logging_options(
          experiment_path='test_experiment',
          logging_path=logging_path,
          global_set=True)
      assert proxai._LOGGING_OPTIONS.logging_path == os.path.join(
          logging_path, 'test_experiment')
      assert proxai._ROOT_LOGGING_PATH == logging_path

  def test_inherit_options(self):
    with tempfile.TemporaryDirectory() as logging_path:
      base_options = types.LoggingOptions(
          logging_path=logging_path,
          stdout=True,
          hide_sensitive_content=True
      )
      result_options, _ = proxai._set_logging_options(
          logging_options=base_options)
      assert result_options.stdout == True
      assert result_options.hide_sensitive_content == True

  def test_creates_experiment_subdirectory(self):
    with tempfile.TemporaryDirectory() as root_path:
      logging_options, _ = proxai._set_logging_options(
          logging_path=root_path,
          experiment_path='new_subdir/new_subdir2'
      )
      expected_path_1 = os.path.join(root_path, 'new_subdir')
      expected_path_2 = os.path.join(expected_path_1, 'new_subdir2')
      assert os.path.exists(expected_path_1)
      assert os.path.exists(expected_path_2)
      assert logging_options.logging_path == expected_path_2

  def test_all_none(self):
    logging_options, root_logging_path = proxai._set_logging_options(
        experiment_path=None,
        logging_path=None,
        logging_options=None
    )
    assert logging_options.logging_path is None
    assert root_logging_path is None

  def test_global_cleanup(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    with tempfile.TemporaryDirectory() as logging_path:
      proxai._set_logging_options(
          logging_path=logging_path,
          global_set=True
      )
      assert proxai._ROOT_LOGGING_PATH == logging_path

    proxai._set_logging_options(
        logging_path=None,
        global_set=True
    )
    assert proxai._ROOT_LOGGING_PATH is None

  def test_default_options(self):
    logging_options, root_logging_path = proxai._set_logging_options()
    assert logging_options.logging_path is None
    assert root_logging_path is None


class TestInitProxdashOptions:
  def test_default_options(self):
    proxdash_options = proxai._set_proxdash_options()
    assert proxdash_options.stdout == False
    assert proxdash_options.hide_sensitive_content == False
    assert proxdash_options.disable_proxdash == False

  def test_custom_options(self):
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    proxdash_options = proxai._set_proxdash_options(
        proxdash_options=base_options)
    assert proxdash_options.stdout == True
    assert proxdash_options.hide_sensitive_content == True
    assert proxdash_options.disable_proxdash == True

  def test_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    proxai._set_proxdash_options(
        proxdash_options=base_options,
        global_set=True)
    assert proxai._PROXDASH_OPTIONS.stdout == True
    assert proxai._PROXDASH_OPTIONS.hide_sensitive_content == True
    assert proxai._PROXDASH_OPTIONS.disable_proxdash == True

  def test_global_init_multiple(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    options_1 = types.ProxDashOptions(stdout=True)
    options_2 = types.ProxDashOptions(hide_sensitive_content=True)

    proxai._set_proxdash_options(proxdash_options=options_1, global_set=True)
    proxai._set_proxdash_options(proxdash_options=options_2, global_set=True)

    assert proxai._PROXDASH_OPTIONS.stdout == False
    assert proxai._PROXDASH_OPTIONS.hide_sensitive_content == True
    assert proxai._PROXDASH_OPTIONS.disable_proxdash == False

  def test_inherit_options(self):
    base_options = types.ProxDashOptions(
        stdout=True,
        hide_sensitive_content=True,
        disable_proxdash=True
    )
    result_options = proxai._set_proxdash_options(proxdash_options=base_options)
    assert result_options.stdout == True
    assert result_options.hide_sensitive_content == True
    assert result_options.disable_proxdash == True


class TestInitAllowMultiprocessing:
  def test_default_value(self):
    assert proxai._set_allow_multiprocessing() is None

  def test_valid_value(self):
    assert proxai._set_allow_multiprocessing(True) == True
    assert proxai._set_allow_multiprocessing(False) == False

  def test_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_allow_multiprocessing(True, global_set=True)
    assert proxai._ALLOW_MULTIPROCESSING == True

  def test_global_init_multiple(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_allow_multiprocessing(True, global_set=True)
    proxai._set_allow_multiprocessing(False, global_set=True)
    assert proxai._ALLOW_MULTIPROCESSING == False

  def test_no_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    original_value = proxai._ALLOW_MULTIPROCESSING
    proxai._set_allow_multiprocessing(True, global_set=False)
    assert proxai._ALLOW_MULTIPROCESSING == original_value


class TestInitFeatureMappingStrategy:
  def test_default_value(self):
    assert proxai._set_feature_mapping_strategy() is None

  def test_valid_value(self):
    assert (proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.STRICT) ==
        types.FeatureMappingStrategy.STRICT)
    assert (proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.BEST_EFFORT) ==
        types.FeatureMappingStrategy.BEST_EFFORT)

  def test_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.STRICT, global_set=True)
    assert (proxai._FEATURE_MAPPING_STRATEGY ==
        types.FeatureMappingStrategy.STRICT)

  def test_global_init_multiple(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.STRICT, global_set=True)
    proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.BEST_EFFORT, global_set=True)
    assert (proxai._FEATURE_MAPPING_STRATEGY ==
        types.FeatureMappingStrategy.BEST_EFFORT)

  def test_no_global_init(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    original_value = proxai._FEATURE_MAPPING_STRATEGY
    proxai._set_feature_mapping_strategy(
        types.FeatureMappingStrategy.STRICT, global_set=False)
    assert proxai._FEATURE_MAPPING_STRATEGY == original_value


class TestRetryIfErrorCached:
  def test_returns_error_from_cache(self):
    proxai.set_run_type(types.RunType.TEST)
    cache_path = tempfile.TemporaryDirectory()
    proxai.connect(cache_path=cache_path.name, allow_multiprocessing=False)
    # First call:
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'
    # Second call:
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.CACHE
    assert response.response_record.error == 'Temp Error'

  def test_makes_provider_call_when_retry_if_error_cached_is_true(self):
    proxai.set_run_type(types.RunType.TEST)
    cache_path = tempfile.TemporaryDirectory()
    proxai.connect(
      cache_path=cache_path.name,
      cache_options=types.CacheOptions(retry_if_error_cached=True),
      allow_multiprocessing=False)
    # First call:
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'
    # Second call:
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'


class TestSuppressProviderErrors:
  def test_connect_with_suppress_provider_errors(self):
    proxai.set_run_type(types.RunType.TEST)

    # Before connect:
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True)

    # After simple connect:
    proxai.connect()
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True)

    # After connect with suppress_provider_errors=True:
    proxai.connect(suppress_provider_errors=True)
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'

    # After connect with suppress_provider_errors=False:
    proxai.connect(suppress_provider_errors=False)
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True)

  def test_generate_text_with_suppress_provider_errors(self):
    proxai.set_run_type(types.RunType.TEST)

    # Before connect:
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True,
          suppress_provider_errors=False)

    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'

    # After simple connect:
    proxai.connect()
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True,
          suppress_provider_errors=False)

    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'

  def test_override_suppress_provider_errors(self):
    proxai.set_run_type(types.RunType.TEST)

    # False override:
    proxai.connect(suppress_provider_errors=True)
    with pytest.raises(Exception):
      proxai.generate_text(
          'hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'),
          extensive_return=True,
          suppress_provider_errors=False)

    # True override:
    proxai.connect(suppress_provider_errors=False)
    response = proxai.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        extensive_return=True,
        suppress_provider_errors=True)
    assert response.response_source == types.ResponseSource.PROVIDER
    assert response.response_record.error == 'Temp Error'


class TestRegisterModel:
  def test_not_supported_provider(self):
    with pytest.raises(ValueError):
      proxai.set_model(
          generate_text=('not_supported_provider', 'not_supported_model'))

  def test_not_supported_model(self):
    with pytest.raises(ValueError):
      proxai.set_model(generate_text=('openai', 'not_supported_model'))

  def test_successful_register_model(self):
    proxai.set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=('openai', 'o1'))
    assert proxai._REGISTERED_MODEL_CONNECTORS[
        types.CallType.GENERATE_TEXT].provider_model == types.ProviderModelType(
            provider='openai',
            model='o1',
            provider_model_identifier='o1-2024-12-17')


class TestGenerateText:
  def _test_generate_text(
      self,
      provider_model: types.ProviderModelIdentifierType):
    model_configs_instance = proxai._get_model_configs()
    provider_model = model_configs_instance.get_provider_model(provider_model)
    proxai.set_run_type(types.RunType.TEST)
    proxai.set_model(generate_text=provider_model)
    assert proxai._REGISTERED_MODEL_CONNECTORS[
        types.CallType.GENERATE_TEXT].provider_model == provider_model

    text = proxai.generate_text('Hello, my name is')
    assert text == 'mock response'
    assert provider_model in proxai._MODEL_CONNECTORS
    assert proxai._MODEL_CONNECTORS[provider_model] is not None

  def test_openai(self):
    self._test_generate_text(('openai', 'gpt-4.1'))

  def test_claude(self):
    self._test_generate_text(('claude', 'opus-4'))

  def test_gemini(self):
    self._test_generate_text(('gemini', 'gemini-3-pro'))

  def test_cohere(self):
    self._test_generate_text(('cohere', 'command-r'))

  def test_databricks(self):
    self._test_generate_text(('databricks', 'llama-4-maverick'))

  def test_mistral(self):
    self._test_generate_text(('mistral', 'open-mistral-7b'))

  def test_huggingface(self):
    self._test_generate_text(('huggingface', 'gemma-2-2b-it'))


class TestDefaultModel:
  def test_default_model_cache_manager_platform(self):
    assert proxai._PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE == True
    assert (
        proxai._DEFAULT_MODEL_CACHE_MANAGER.cache_options
        .model_cache_duration == 4 * 60 * 60)

  def test_default_model_cache_manager_tempfile(self):
    assert proxai._PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE == True
    cache_dir = platformdirs.PlatformDirs(
        appname="proxai", appauthor="proxai").user_cache_dir
    platform_cache_path = os.path.join(
        cache_dir, model_cache.AVAILABLE_MODELS_PATH)

    # First, corrupt platform cache file:
    with open(platform_cache_path, 'w') as f:
      f.write('invalid_json')

    # Second, init default model cache manager:
    proxai._init_default_model_cache_manager()

    # Third, check that the default model cache manager is loaded from tempfile:
    assert proxai._PLATFORM_USED_FOR_DEFAULT_MODEL_CACHE == False
    assert (
        proxai._DEFAULT_MODEL_CACHE_MANAGER.cache_options
        .model_cache_duration is None)

    # Finally, remove invalid json file:
    os.remove(platform_cache_path)

  def test_reset_platform_cache(self):
    cache_dir = platformdirs.PlatformDirs(
        appname="proxai", appauthor="proxai").user_cache_dir
    platform_cache_path = os.path.join(
        cache_dir, model_cache.AVAILABLE_MODELS_PATH)
    with open(platform_cache_path, 'w') as f:
      f.write('{}')

    assert os.path.exists(platform_cache_path)
    proxai.reset_platform_cache()
    assert not os.path.exists(platform_cache_path)


class TestConnectProxdashConnection:
  def test_connect_proxdash_connection(self, requests_mock, monkeypatch):
    # Setup
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_proxdash_api_key')
    requests_mock.get(
        'https://proxainest-production.up.railway.app/ingestion/verify-key',
        text='{"success": true, "data": {"permission": "ALL"}}',
        status_code=200,
    )

    # First connection
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect()
    assert proxai._PROXDASH_CONNECTION is not None
    first_connection = proxai._PROXDASH_CONNECTION
    assert (
        first_connection.status == types.ProxDashConnectionStatus.CONNECTED)

    # Second connection should reuse existing connection
    proxai.connect()
    assert len(requests_mock.request_history) == 1

  def test_connect_proxdash_connection_disabled(
      self, monkeypatch, requests_mock):
    # Setup
    monkeypatch.setenv('PROXDASH_API_KEY', 'test_proxdash_api_key')
    requests_mock.get(
        'https://proxainest-production.up.railway.app/ingestion/verify-key',
        text='{"success": true, "data": {"permission": "ALL"}}',
        status_code=200,
    )

    # First connection with disabled proxdash
    proxai.set_run_type(types.RunType.TEST)
    proxai.connect(
        proxdash_options=types.ProxDashOptions(disable_proxdash=True))
    assert proxai._PROXDASH_CONNECTION is not None
    first_connection = proxai._PROXDASH_CONNECTION
    assert first_connection.status == types.ProxDashConnectionStatus.DISABLED
    # No connection attempts when disabled
    assert len(requests_mock.request_history) == 0

    # Second connection should reuse existing connection but this time
    # it should make the api validity call
    proxai.connect()
    assert len(requests_mock.request_history) == 1

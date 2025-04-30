import os
import tempfile
import time
import json
import pytest
import proxai as px
import proxai.connectors.model_configs as model_configs


class TestProxaiApiUseCases:
  @pytest.fixture(autouse=True)
  def setup_test(self, monkeypatch):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
      for api_key in api_key_list:
        monkeypatch.setenv(api_key, 'test_api_key')
    monkeypatch.delenv('MOCK_SLOW_PROVIDER', raising=False)
    self.temp_paths = {}
    px.reset_state()
    px.set_run_type(px.types.RunType.TEST)
    px.models.allow_multiprocessing = False
    yield

  def _get_path_dir(self, temp_path: str):
    if temp_path in self.temp_paths:
      path = os.path.join(self.temp_paths[temp_path].name, temp_path)
    else:
      self.temp_paths[temp_path] = tempfile.TemporaryDirectory()
      path = os.path.join(self.temp_paths[temp_path].name, temp_path)
    # Create the subdirectory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    return path

  def _test_uncached_get_all_models(self):
    # Check that model cache is not created yet:
    assert px.models.model_cache_manager.get(
        'GENERATE_TEXT').working_models == set()
    assert not os.path.exists(px.models.model_cache_manager.cache_path)

    # If multiprocessing is disabled, it should be fast:
    start = time.time()
    px.models.allow_multiprocessing = False
    px.models.list_models()
    px.models.allow_multiprocessing = None
    total_time = time.time() - start
    assert total_time < 1

    # Check that model cache is created in the cache path:
    assert os.path.exists(px.models.model_cache_manager.cache_path)
    with open(px.models.model_cache_manager.cache_path, 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['failed_models']) == 1
      assert data['GENERATE_TEXT']['unprocessed_models'] == []
      assert data['GENERATE_TEXT']['filtered_models'] == []
      assert len(data['GENERATE_TEXT']['working_models']) >= 30
      assert len(data['GENERATE_TEXT']['provider_queries']) >= 30

  def _test_cached_get_all_models(self):
    # Check that model cache is created:
    assert len(px.models.model_cache_manager.get(
        'GENERATE_TEXT').working_models) > 30
    assert os.path.exists(px.models.model_cache_manager.cache_path)
    with open(px.models.model_cache_manager.cache_path, 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['failed_models']) == 1
      assert data['GENERATE_TEXT']['unprocessed_models'] == []
      assert data['GENERATE_TEXT']['filtered_models'] == []
      assert len(data['GENERATE_TEXT']['working_models']) >= 30
      assert len(data['GENERATE_TEXT']['provider_queries']) >= 30

    # Even multiprocessing is enabled, if results are cached, it should be
    # fast:
    start = time.time()
    px.models.allow_multiprocessing = True
    px.models.list_models()
    px.models.allow_multiprocessing = None
    total_time = time.time() - start
    assert total_time < 1

    # Check that model cache in cache path is not changed:
    assert os.path.exists(px.models.model_cache_manager.cache_path)
    with open(px.models.model_cache_manager.cache_path, 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['failed_models']) == 1
      assert data['GENERATE_TEXT']['unprocessed_models'] == []
      assert data['GENERATE_TEXT']['filtered_models'] == []
      assert len(data['GENERATE_TEXT']['working_models']) >= 30
      assert len(data['GENERATE_TEXT']['provider_queries']) >= 30

  def test_generate_text(self):
    text = px.generate_text('hello')
    assert text == 'mock response'

    logging_record = px.generate_text(
        prompt='hello',
        provider_model=px.models.get_model(
            'claude', 'haiku', clear_model_cache=True),
        extensive_return=True)
    assert logging_record.response_record.response == 'mock response'
    assert logging_record.query_record.provider_model.model == 'haiku'
    assert logging_record.response_source == px.types.ResponseSource.PROVIDER

    logging_record = px.generate_text(
        prompt='hello',
        system='You are a helpful assistant.',
        max_tokens=100,
        temperature=0.5,
        stop=['\n\n'],
        provider_model=('openai', 'gpt-3.5-turbo'),
        use_cache=False,
        unique_response_limit=1,
        extensive_return=True)
    assert logging_record.query_record.prompt == 'hello'
    assert logging_record.query_record.system == 'You are a helpful assistant.'
    assert logging_record.query_record.max_tokens == 100
    assert logging_record.query_record.temperature == 0.5
    assert logging_record.query_record.stop == ['\n\n']
    assert logging_record.query_record.provider_model.model == 'gpt-3.5-turbo'
    assert logging_record.response_record.response == 'mock response'
    assert logging_record.response_source == px.types.ResponseSource.PROVIDER

  def test_generate_text_with_use_cache_before_connect(self):
    with pytest.raises(ValueError):
      px.generate_text(use_cache=True)

  def test_models_get_all_models(self):
    start = time.time()
    models = px.models.list_models(clear_model_cache=True)
    assert len(models) > 15
    assert time.time() - start < 1

    start = time.time()
    models = px.models.list_models(only_largest_models=True)
    assert len(models) < 15
    assert time.time() - start < 1

  def test_models_get_all_models_with_multiprocessing_and_model_test_timeout(
      self, monkeypatch):
    monkeypatch.setenv('MOCK_SLOW_PROVIDER', 'test_api_key')
    start = time.time()
    px.models.allow_multiprocessing = True
    px.models.model_test_timeout = 2
    models = px.models.list_models(
        return_all=True,
        clear_model_cache=True)
    px.models.allow_multiprocessing = None
    px.models.model_test_timeout = 25
    assert len(models.working_models) > 15
    assert (
        model_configs.ALL_MODELS['mock_slow_provider']['mock_slow_model']
        in models.failed_models)
    assert time.time() - start < 5

  def test_models_apis(self):
    px.connect(cache_options=px.CacheOptions(clear_model_cache_on_connect=True))

    self._test_uncached_get_all_models()

    # --- get_providers ---
    # This should be fast because of model cache:
    start = time.time()
    providers = px.models.list_providers()
    assert len(providers) > 5
    assert time.time() - start < 1

    # --- get_provider_models ---
    # This should be fast because of model cache:
    start = time.time()
    models = px.models.list_provider_models('openai')
    assert len(models) > 2
    assert time.time() - start < 1

    # --- get_provider_model ---
    # This should be fast because of model cache:
    start = time.time()
    provider_model = px.models.get_model('openai', 'gpt-4')
    assert provider_model.provider == 'openai'
    assert provider_model.model == 'gpt-4'
    assert time.time() - start < 1

    # --- get_all_models with largest models ---
    start = time.time()
    models = px.models.list_models(only_largest_models=True)
    assert len(models) < 15
    assert time.time() - start < 1

    # --- get_all_models with clear_model_cache ---
    start = time.time()
    px.models.allow_multiprocessing = False
    models = px.models.list_models(clear_model_cache=True)
    px.models.allow_multiprocessing = None
    assert len(models) > 15
    assert time.time() - start < 1

  def test_set_model(self):
    px.models.list_models(clear_model_cache=True)

    # Test default model
    px.set_model(('claude', 'haiku'))
    logging_record = px.generate_text('hello', extensive_return=True)
    assert logging_record.query_record.provider_model.model == 'haiku'

    # Test setting model with generate_text parameter
    px.set_model(generate_text=('openai', 'gpt-4'))
    logging_record = px.generate_text('hello', extensive_return=True)
    assert logging_record.query_record.provider_model.model == 'gpt-4'

    # Test setting model with provider_model parameter
    px.set_model(provider_model=('openai', 'gpt-3.5-turbo'))
    logging_record = px.generate_text('hello', extensive_return=True)
    assert logging_record.query_record.provider_model.model == 'gpt-3.5-turbo'

    # Test setting model with provider_model from get_provider_model
    px.set_model(px.models.get_model('claude', 'haiku'))
    logging_record = px.generate_text('hello', extensive_return=True)
    assert logging_record.query_record.provider_model.model == 'haiku'

    # Test error when both parameters are set
    with pytest.raises(
        ValueError,
        match='provider_model and generate_text cannot be set at the same time'
    ):
      px.set_model(
          provider_model=('openai', 'gpt-4'),
          generate_text=('openai', 'gpt-3.5-turbo'))

    # Test error when neither parameter is set
    with pytest.raises(
        ValueError, match='provider_model or generate_text must be set'):
      px.set_model()

  def test_model_cache_with_different_connect_cache_paths(self):
    # --- Default model cache directory test before connect ---
    # First call:
    self._test_uncached_get_all_models()
    # Second call should be fast because of model cache:
    self._test_cached_get_all_models()

    # --- First connect without cache path ---
    px.connect()
    # Call should be fast because still using default model cache:
    self._test_cached_get_all_models()

    # --- Second connect with new cache_path ---
    cache_path = self._get_path_dir(
        'test_model_cache_with_different_connect_cache_paths_cache_path')
    px.connect(cache_path=cache_path)
    # First call:
    self._test_uncached_get_all_models()
    # Second call should be fast because of model cache:
    self._test_cached_get_all_models()

    # --- Third connect with same cache_path ---
    px.connect(cache_path=cache_path)
    # Call should be fast because of the same cache path:
    self._test_cached_get_all_models()

    # --- Fourth connect with new cache_path_2 ---
    cache_path_2 = self._get_path_dir(
        'test_model_cache_with_different_connect_cache_paths_cache_path_2')
    px.connect(cache_path=cache_path_2)
    # First call:
    self._test_uncached_get_all_models()
    # Second call should be fast because of model cache:
    self._test_cached_get_all_models()

    # --- Fifth connect with same cache path ---
    px.connect(cache_path=cache_path)
    # Call should be fast because of the previously used cache path:
    self._test_cached_get_all_models()

    # --- Sixth connect with default cache path ---
    px.connect()
    # Call should be fast because using default model cache:
    self._test_cached_get_all_models()

  def test_query_cache_with_different_connect_cache_paths(self):
    # --- Before connect ---
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER

    # --- Connect cache path 1 ---
    cache_path_1 = self._get_path_dir('cache_path_1')
    px.connect(cache_path=cache_path_1)
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Connect cache path 2 ---
    cache_path_2 = self._get_path_dir('cache_path_2')
    px.connect(cache_path=cache_path_2)
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Connect cache path 1 again ---
    cache_path_1 = self._get_path_dir('cache_path_1')
    px.connect(cache_path=cache_path_1)
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Connect default ---
    px.connect()
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER

  def test_query_cache_with_different_connect_cache_options(self):
    # --- Connect cache path ---
    cache_path = self._get_path_dir('cache_path')
    px.connect(cache_path=cache_path)
    # First call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Second call:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Connect cache path from CacheOptions ---
    px.connect(
        cache_options=px.CacheOptions(cache_path=cache_path))
    # Still same cache path, so still cache:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Increase unique_response_limit to 3 ---
    px.connect(
        cache_path=cache_path,
        cache_options=px.CacheOptions(unique_response_limit=3))
    # This is second actual provider call, so provider:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # This is third actual provider call, so provider:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # There is already 3 provider calls, so this should be from cache:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Clear query cache on connect ---
    px.connect(
        cache_path=cache_path,
        cache_options=px.CacheOptions(
            unique_response_limit=3,
            clear_query_cache_on_connect=True))
    # This is first actual provider call after clearing cache, so provider:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # This is second actual provider call, so provider:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # This is third actual provider call, so provider:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # There is already 3 provider calls, so this should be from cache:
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

  def test_query_cache_with_different_generate_text_options(self):
    cache_path = self._get_path_dir('cache_path')
    px.connect(cache_path=cache_path)

    # --- First call ---
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER

    # --- Second call ---
    response = px.generate_text('hello', extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Third call without cache ---
    response = px.generate_text(
        'hello',
        use_cache=False,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER

    # --- Increase unique_response_limit to 3 ---
    # Second provider call:
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Third provider call:
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    # Already 3 provider calls, so this should be from cache:
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

  def test_query_cache_override_options(self):
    cache_path = self._get_path_dir('cache_path')
    px.connect(cache_options=px.CacheOptions(
        cache_path=cache_path,
        unique_response_limit=2))

    # --- No cache ---
    response = px.generate_text(
        'hello',
        use_cache=False,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    response = px.generate_text(
        'hello',
        use_cache=False,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER

    # --- Set unique_response_limit to 1 ---
    response = px.generate_text(
        'hello',
        unique_response_limit=1,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    response = px.generate_text(
        'hello',
        unique_response_limit=1,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

    # --- Set unique_response_limit to 3 ---
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.PROVIDER
    response = px.generate_text(
        'hello',
        unique_response_limit=3,
        extensive_return=True)
    assert response.response_source == px.types.ResponseSource.CACHE

  def test_query_cache_retry_if_error_cached(self):
    # This test is passed to tests/test_proxai.py because it requires
    # private variables and not feasible for API test. Please refer to
    # test_retry_if_error_cached in tests/test_proxai.py for more details.
    pass

  def test_connect_with_different_experiment_paths(self):
    logging_path = self._get_path_dir('logging_path')
    px.connect(
        experiment_path='experiment_path_1',
        logging_path=logging_path)
    px.generate_text('hello')
    with open(os.path.join(
        logging_path, 'experiment_path_1/provider_queries.log'), 'r') as f:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello'

    px.connect(
        experiment_path='experiment_dir_1/experiment_path_2',
        logging_path=logging_path)
    px.generate_text('hello2')
    with open(os.path.join(
        logging_path,
        'experiment_dir_1/experiment_path_2/provider_queries.log'), 'r') as f:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello2'

    # Test even experiment path conflict works:
    px.connect(
        experiment_path='experiment_path_1/experiment_path_3',
        logging_path=logging_path)
    px.generate_text('hello3')
    with open(os.path.join(
        logging_path,
        'experiment_path_1/experiment_path_3/provider_queries.log'), 'r') as f:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello3'

  def test_connect_with_different_logging_paths(self):
    logging_path = self._get_path_dir('logging_path')
    cache_path = self._get_path_dir('cache_path')
    px.connect(
        logging_path=logging_path,
        cache_path=cache_path)
    px.generate_text('hello')
    with open(os.path.join(logging_path, 'provider_queries.log'), 'r') as f:
      # First logging record:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello'
      assert log_record['response_source'] == px.types.ResponseSource.CACHE
      assert (
          log_record['look_fail_reason'] ==
          px.types.CacheLookFailReason.CACHE_NOT_FOUND)
      # Second logging record:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello'
      assert log_record['response_record']['response'] == 'mock response'
      assert log_record['response_source'] == px.types.ResponseSource.PROVIDER
      # End of file:
      assert f.readline() == ''

    logging_path_2 = self._get_path_dir('logging_path_2')
    px.connect(
        logging_path=logging_path_2,
        cache_path=cache_path,
        allow_multiprocessing=False)
    px.generate_text('hello')
    with open(os.path.join(logging_path_2, 'provider_queries.log'), 'r') as f:
      # First logging record:
      log_record = json.loads(f.readline())
      assert log_record['query_record']['prompt'] == 'hello'
      assert log_record['response_record']['response'] == 'mock response'
      assert log_record['response_source'] == px.types.ResponseSource.CACHE
      # End of file:
      assert f.readline() == ''

  def test_connect_with_different_logging_options(self):
    logging_path = self._get_path_dir('logging_path')
    px.connect(
        logging_options=px.LoggingOptions(
            logging_path=logging_path,
            hide_sensitive_content=True))
    px.generate_text('hello')
    with open(os.path.join(logging_path, 'provider_queries.log'), 'r') as f:
      # First logging record:
      log_record = json.loads(f.readline())
      assert (
          log_record['query_record']['prompt'] ==
          '<sensitive content hidden>')
      assert (
          log_record['response_record']['response'] ==
          '<sensitive content hidden>')
      assert log_record['response_source'] == px.types.ResponseSource.PROVIDER
      # End of file:
      assert f.readline() == ''

  def test_connect_with_strict_feature_test(self):
    px.connect(strict_feature_test=False)
    px.generate_text(
        'hello',
        stop='STOP',
        provider_model=('mistral', 'mistral-large'))

    px.connect(strict_feature_test=True)
    with pytest.raises(Exception):
      px.generate_text(
          'hello',
          stop='STOP',
          provider_model=('mistral', 'mistral-large'))

  def test_get_current_options(self):
    options = px.get_current_options()
    assert options.run_type == px.types.RunType.TEST
    assert options.logging_options.logging_path == None
    assert options.logging_options.stdout == False
    assert options.logging_options.hide_sensitive_content == False
    assert options.cache_options.cache_path == None
    assert options.cache_options.unique_response_limit == 1
    assert options.cache_options.model_cache_duration == None
    assert options.cache_options.retry_if_error_cached == False
    assert options.cache_options.clear_query_cache_on_connect == False
    assert options.cache_options.clear_model_cache_on_connect == False
    assert options.proxdash_options.stdout == False
    assert options.proxdash_options.hide_sensitive_content == False
    assert options.proxdash_options.disable_proxdash == False
    assert options.allow_multiprocessing == True
    assert options.model_test_timeout == 25
    assert options.strict_feature_test == False

    logging_path = self._get_path_dir('logging_path')
    cache_path = self._get_path_dir('cache_path')
    px.connect(
        logging_options=px.LoggingOptions(
            logging_path=logging_path,
            stdout=True,
            hide_sensitive_content=True),
        cache_options=px.CacheOptions(
            cache_path=cache_path,
            unique_response_limit=2,
            retry_if_error_cached=True,
            clear_query_cache_on_connect=True,
            clear_model_cache_on_connect=True),
        proxdash_options=px.ProxDashOptions(
            stdout=True,
            hide_sensitive_content=True,
            disable_proxdash=True),
        allow_multiprocessing=False,
        model_test_timeout=45,
        strict_feature_test=True)
    options = px.get_current_options()
    assert options.run_type == px.types.RunType.TEST
    assert options.logging_options.logging_path == logging_path
    assert options.logging_options.stdout == True
    assert options.logging_options.hide_sensitive_content == True
    assert options.cache_options.cache_path == cache_path
    assert options.cache_options.unique_response_limit == 2
    assert options.cache_options.retry_if_error_cached == True
    assert options.cache_options.clear_query_cache_on_connect == True
    assert options.cache_options.clear_model_cache_on_connect == True
    assert options.proxdash_options.stdout == True
    assert options.proxdash_options.hide_sensitive_content == True
    assert options.proxdash_options.disable_proxdash == True
    assert options.allow_multiprocessing == False
    assert options.model_test_timeout == 45
    assert options.strict_feature_test == True

    options = px.get_current_options(json=True)
    assert options['run_type'] == px.types.RunType.TEST.value
    assert options['logging_options']['logging_path'] == logging_path
    assert options['logging_options']['stdout'] == True
    assert options['logging_options']['hide_sensitive_content'] == True
    assert options['cache_options']['cache_path'] == cache_path
    assert options['cache_options']['unique_response_limit'] == 2
    assert options['cache_options']['retry_if_error_cached'] == True
    assert options['cache_options']['clear_query_cache_on_connect'] == True
    assert options['cache_options']['clear_model_cache_on_connect'] == True
    assert options['proxdash_options']['stdout'] == True
    assert options['proxdash_options']['hide_sensitive_content'] == True
    assert options['proxdash_options']['disable_proxdash'] == True

  def test_get_summary(self):
    cache_path = self._get_path_dir('cache_path')
    px.connect(cache_path=cache_path)
    px.generate_text('hello')
    px.generate_text('hello')
    px.generate_text('hello')

    summary = px.get_summary()
    assert summary.provider_stats.total_queries == 1
    assert summary.cache_stats.total_cache_hit == 2
    assert summary.providers['openai'].provider_stats.total_queries == 1
    assert summary.providers['openai'].cache_stats.total_cache_hit == 2

    summary = px.get_summary(json=True)
    assert summary['provider_stats']['total_queries'] == 1
    assert summary['cache_stats']['total_cache_hit'] == 2
    assert summary['providers']['openai']['provider_stats'][
        'total_queries'] == 1
    assert summary['providers']['openai']['cache_stats']['total_cache_hit'] == 2

  def test_check_health(self):
    model_status = px.check_health(
        extensive_return=True,
        allow_multiprocessing=False)
    assert len(model_status.working_models) > 10
    assert len(model_status.failed_models) == 1

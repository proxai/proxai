import os
import tempfile
import time
import json
from typing import Dict, Optional
import pytest
import proxai as px


class TestProxaiApiUseCases:
  @pytest.fixture(autouse=True)
  def setup_test(self):
    self.temp_paths = {}
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

  def test_generate_text(self):
    text = px.generate_text('hello')
    assert text == 'mock response'

  def test_generate_text_with_all_options(self):
    text = px.generate_text(
        prompt='hello',
        system='You are a helpful assistant.',
        max_tokens=100,
        temperature=0.5,
        stop=['\n\n'],
        provider='openai',
        model='gpt-4',
        use_cache=False,
        unique_response_limit=1,
        extensive_return=True)
    assert text.response_record.response == 'mock response'
    assert text.response_source == px.types.ResponseSource.PROVIDER

  def test_generate_text_with_use_cache_before_connect(self):
    with pytest.raises(ValueError):
      px.generate_text(use_cache=True)

  def test_models_generate_text(self):
    models = px.models.generate_text()
    assert len(models) > 10

  def test_models_generate_text_only_largest_models_multiprocessing(self):
    px.models.allow_multiprocessing = True
    models = px.models.generate_text(only_largest_models=True)
    assert len(models) < 10

  def test_set_model(self):
    for provider, model in px.models.generate_text(only_largest_models=True):
      px.set_model(generate_text=(provider, model))
      assert px.generate_text('hello') == 'mock response'

  def test_model_cache_with_different_connect_cache_paths(self):
    # --- Default model cache directory test ---
    # First call:
    px.models.generate_text()
    # Second call should be faster because of model cache:
    start = time.time()
    px.models.allow_multiprocessing = True
    px.models.generate_text()
    px.models.allow_multiprocessing = False
    total_time = time.time() - start
    assert total_time < 1

    # --- First connect without cache path ---
    px.connect(allow_multiprocessing=False)
    # First call:
    px.models.generate_text()
    # Second call should be faster because of model cache:
    start = time.time()
    px.models.allow_multiprocessing = True
    px.models.generate_text()
    px.models.allow_multiprocessing = False
    total_time = time.time() - start
    assert total_time < 1

    def _check_model_cache_path(
        cache_path: str,
        min_expected_records: int):
      assert os.path.exists(os.path.join(cache_path, 'available_models.json'))
      with open(os.path.join(cache_path, 'available_models.json'), 'r') as f:
        data = json.load(f)
        assert data['GENERATE_TEXT']['failed_models'] == []
        assert data['GENERATE_TEXT']['unprocessed_models'] == []
        assert data['GENERATE_TEXT']['filtered_models'] == []
        assert (
            len(data['GENERATE_TEXT']['working_models']) >=
            min_expected_records)
        assert (
            len(data['GENERATE_TEXT']['provider_queries']) >=
            min_expected_records)

    # --- First connect ---
    cache_path = self._get_path_dir('cache_path')
    px.connect(
        cache_path=cache_path,
        allow_multiprocessing=False)
    # Cache file is not created yet because nothing saved to ModelCacheManager.
    assert not os.path.exists(os.path.join(cache_path, 'available_models.json'))
    px.models.generate_text(only_largest_models=True)
    # Cache file is created because some models are saved to ModelCacheManager.
    _check_model_cache_path(cache_path, min_expected_records=5)

    # --- Second connect with same cache path ---
    px.connect(
        cache_path=cache_path,
        allow_multiprocessing=True)
    # Cache file is still there because same cache path.
    _check_model_cache_path(cache_path, min_expected_records=5)
    px.models.generate_text(only_largest_models=False)
    # Cache file is updated and more models are saved to ModelCacheManager
    # because only_largest_models is False.
    _check_model_cache_path(cache_path, min_expected_records=30)

    # --- Third connect with different cache path ---
    cache_path_2 = self._get_path_dir('cache_path_2')
    # Cache file is not created yet because nothing saved to ModelCacheManager
    # and this cache path is not used in previous connect.
    assert not os.path.exists(
        os.path.join(cache_path_2, 'available_models.json'))
    px.connect(
        cache_path=cache_path_2,
        allow_multiprocessing=False)
    px.models.generate_text(only_largest_models=True)
    # Cache file is created because some models are saved to ModelCacheManager.
    _check_model_cache_path(cache_path_2, min_expected_records=5)

    # --- Fourth connect with same cache path ---
    px.connect(
        cache_path=cache_path,
        allow_multiprocessing=True)
    # Cache file is still there because same cache path.
    _check_model_cache_path(cache_path, min_expected_records=30)
    px.models.generate_text(only_largest_models=True)
    # Nothing changed because same cache path and same only_largest_models.
    _check_model_cache_path(cache_path, min_expected_records=30)

  def test_model_cache_with_different_model_generate_text_options(self):
    cache_path = self._get_path_dir('cache_path')
    px.connect(cache_path=cache_path, allow_multiprocessing=False)

    # At the first call, more than 30 models are saved to cache:
    px.models.generate_text(only_largest_models=False)
    with open(os.path.join(cache_path, 'available_models.json'), 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['working_models']) > 10
      assert len(data['GENERATE_TEXT']['provider_queries']) > 10

    # At the second call, only largest models called but because other models
    # are already cached, so it should have same number of records:
    px.models.generate_text(only_largest_models=True)
    with open(os.path.join(cache_path, 'available_models.json'), 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['working_models']) > 10
      assert len(data['GENERATE_TEXT']['provider_queries']) > 10

    # At the third call, clear_model_cache is True, so it should have less than
    # 10 records because previous models are deleted and only largest models
    # are called:
    px.models.generate_text(
        only_largest_models=True,
        clear_model_cache=True)
    with open(os.path.join(cache_path, 'available_models.json'), 'r') as f:
      data = json.load(f)
      assert len(data['GENERATE_TEXT']['working_models']) > 0
      assert len(data['GENERATE_TEXT']['working_models']) < 10
      assert len(data['GENERATE_TEXT']['provider_queries']) > 0
      assert len(data['GENERATE_TEXT']['provider_queries']) < 10

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
        system='You are a helpful assistant.',
        provider=px.types.Provider.HUGGING_FACE,
        model=px.types.HuggingFaceModel.GOOGLE_GEMMA_7B_IT)

    px.connect(strict_feature_test=True)
    with pytest.raises(Exception):
      px.generate_text(
          'hello',
          system='You are a helpful assistant.',
          provider=px.types.Provider.HUGGING_FACE,
          model=px.types.HuggingFaceModel.GOOGLE_GEMMA_7B_IT)

  def test_get_current_options(self):
    options = px.get_current_options()
    assert options.run_type == px.types.RunType.TEST
    assert options.logging_options.logging_path == None
    assert options.logging_options.stdout == False
    assert options.logging_options.hide_sensitive_content == False
    assert options.cache_options.cache_path == None
    assert options.cache_options.unique_response_limit == 1
    assert options.cache_options.duration == None
    assert options.cache_options.retry_if_error_cached == False
    assert options.cache_options.clear_query_cache_on_connect == False
    assert options.cache_options.clear_model_cache_on_connect == False
    assert options.proxdash_options.stdout == False
    assert options.proxdash_options.hide_sensitive_content == False
    assert options.proxdash_options.disable_proxdash == False
    assert options.allow_multiprocessing == True
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
    model_status = px.check_health(detailed=True)
    assert len(model_status.working_models) > 10
    assert len(model_status.failed_models) == 0

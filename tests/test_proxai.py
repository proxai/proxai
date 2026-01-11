"""Tests for user-facing proxai API (px.*)

These tests focus on user workflows and use cases, testing the public API
that users interact with (px.connect, px.generate_text, px.set_model, etc.).
"""
import os
import tempfile
import pytest
import proxai as px
import proxai.types as types
import proxai.connectors.model_configs as model_configs


@pytest.fixture(autouse=True)
def setup_test(monkeypatch):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.setenv(api_key, 'test_api_key')
  px.reset_state()
  _set_test_mode()
  px.set_model(('mock_provider', 'mock_model'))
  yield
  px.reset_state()


def _set_test_mode():
  """Set RunType.TEST on available_models_instance to use mock providers."""
  px.get_default_proxai_client()._available_models_instance.run_type = (
      types.RunType.TEST)


def _get_temp_dir(name: str):
  temp_dir = tempfile.TemporaryDirectory()
  path = os.path.join(temp_dir.name, name)
  os.makedirs(path, exist_ok=True)
  return path, temp_dir


class TestGenerateText:
  """Test px.generate_text() - the core user workflow."""

  def test_generate_text_with_prompt(self):
    response = px.generate_text('hello')
    assert response == 'mock response'

  def test_generate_text_with_messages(self):
    response = px.generate_text(
        messages=[{'role': 'user', 'content': 'hello'}])
    assert response == 'mock response'

  def test_generate_text_with_system_prompt(self):
    response = px.generate_text(
        prompt='hello',
        system='You are a helpful assistant.')
    assert response == 'mock response'

  def test_generate_text_with_provider_model(self):
    response = px.generate_text(
        prompt='hello',
        provider_model=('mock_provider', 'mock_model'))
    assert response == 'mock response'

  def test_generate_text_with_extensive_return(self):
    response = px.generate_text(
        prompt='hello',
        extensive_return=True)
    assert isinstance(response, types.LoggingRecord)
    assert response.response_record.response.value == 'mock response'
    assert response.response_source == types.ResponseSource.PROVIDER

  def test_generate_text_error_raises_exception(self):
    with pytest.raises(Exception, match='Temp Error'):
      px.generate_text(
          prompt='hello',
          provider_model=('mock_failing_provider', 'mock_failing_model'))

  def test_generate_text_with_suppress_provider_errors(self):
    response = px.generate_text(
        prompt='hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True)
    assert response == 'Temp Error'


class TestSetModel:
  """Test px.set_model() - setting default model for subsequent calls."""

  def test_set_model_with_tuple(self):
    px.set_model(('mock_provider', 'mock_model'))
    response = px.generate_text('hello', extensive_return=True)
    assert response.query_record.provider_model.provider == 'mock_provider'
    assert response.query_record.provider_model.model == 'mock_model'

  def test_set_model_with_provider_model_param(self):
    px.set_model(provider_model=('mock_provider', 'mock_model'))
    response = px.generate_text('hello', extensive_return=True)
    assert response.query_record.provider_model.provider == 'mock_provider'

  def test_set_model_with_generate_text_param(self):
    px.set_model(generate_text=('mock_provider', 'mock_model'))
    response = px.generate_text('hello', extensive_return=True)
    assert response.query_record.provider_model.provider == 'mock_provider'

  def test_set_model_persists_for_subsequent_calls(self):
    px.set_model(('mock_provider', 'mock_model'))
    response1 = px.generate_text('hello', extensive_return=True)
    response2 = px.generate_text('world', extensive_return=True)
    assert response1.query_record.provider_model.model == 'mock_model'
    assert response2.query_record.provider_model.model == 'mock_model'

  def test_set_model_can_be_changed(self):
    px.set_model(('mock_provider', 'mock_model'))
    response1 = px.generate_text('hello', extensive_return=True)
    assert response1.query_record.provider_model.model == 'mock_model'

    px.set_model(('mock_failing_provider', 'mock_failing_model'))
    response2 = px.generate_text(
        'hello', suppress_provider_errors=True, extensive_return=True)
    assert response2.query_record.provider_model.model == 'mock_failing_model'


class TestConnect:
  """Test px.connect() - connection and configuration."""

  def test_connect_with_cache_options_enables_caching(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)
    px.set_model(('mock_provider', 'mock_model'))

    response1 = px.generate_text('hello', extensive_return=True)
    response2 = px.generate_text('hello', extensive_return=True)

    assert response1.response_source == types.ResponseSource.PROVIDER
    assert response2.response_source == types.ResponseSource.CACHE

  def test_connect_with_experiment_path(self):
    logging_path, _ = _get_temp_dir('logs')
    px.connect(
        experiment_path='my/experiment',
        logging_options=types.LoggingOptions(logging_path=logging_path))

    options = px.get_current_options()
    assert options.experiment_path == 'my/experiment'
    assert options.logging_options.logging_path == os.path.join(
        logging_path, 'my/experiment')

  def test_connect_resets_previous_configuration(self):
    cache_path1, _ = _get_temp_dir('cache1')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path1))
    options1 = px.get_current_options()

    cache_path2, _ = _get_temp_dir('cache2')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path2))
    options2 = px.get_current_options()

    assert options1.cache_options.cache_path == cache_path1
    assert options2.cache_options.cache_path == cache_path2

  def test_connect_with_suppress_provider_errors(self):
    px.connect(suppress_provider_errors=True)
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)

    response = px.generate_text(
        prompt='hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'))
    assert response == 'Temp Error'

  def test_connect_with_feature_mapping_strategy(self):
    px.connect(feature_mapping_strategy=types.FeatureMappingStrategy.STRICT)
    options = px.get_current_options()
    assert options.feature_mapping_strategy == types.FeatureMappingStrategy.STRICT


class TestModelsApiListModels:
  """Test px.models.list_models() and related functions."""

  def test_list_models_returns_available_models(self):
    models = px.models.list_models()
    assert len(models) > 0
    assert all(hasattr(m, 'provider') for m in models)
    assert all(hasattr(m, 'model') for m in models)

  def test_list_models_with_model_size_filter(self):
    small_models = px.models.list_models(model_size='small')
    large_models = px.models.list_models(model_size='large')
    assert len(small_models) > 0
    assert len(large_models) > 0
    assert len(small_models) != len(large_models)

  def test_list_models_with_model_size_enum(self):
    models = px.models.list_models(model_size=types.ModelSizeType.MEDIUM)
    assert len(models) > 0

  def test_get_model_returns_specific_model(self):
    model = px.models.get_model(provider='mock_provider', model='mock_model')
    assert model.provider == 'mock_provider'
    assert model.model == 'mock_model'

  def test_list_providers_returns_providers(self):
    providers = px.models.list_providers()
    assert len(providers) > 0
    assert any(p in providers for p in ['openai', 'claude', 'gemini'])

  def test_list_provider_models_returns_models_for_provider(self):
    models = px.models.list_provider_models('openai')
    assert len(models) > 0
    assert all(m.provider == 'openai' for m in models)


class TestModelsApiListWorkingModels:
  """Test px.models.list_working_models() and related functions.

  Note: list_working_models() tests all configured models, which takes time.
  Tests are designed to minimize redundant calls by caching results.
  """

  def test_list_working_models_and_related_apis(self):
    # Call list_working_models once with return_all to get full ModelStatus
    result = px.models.list_working_models(return_all=True)

    # Verify list_working_models returns working models
    assert hasattr(result, 'working_models')
    assert hasattr(result, 'failed_models')
    assert len(result.working_models) > 0

    # Verify list_working_providers returns providers
    providers = px.models.list_working_providers()
    assert len(providers) > 0
    all_providers = px.models.list_providers()
    assert all(p in all_providers for p in providers)

    # Verify get_working_model works for a known working model
    first_model = list(result.working_models)[0]
    model = px.models.get_working_model(
        provider=first_model.provider, model=first_model.model)
    assert model.provider == first_model.provider
    assert model.model == first_model.model

    # Verify list_working_provider_models returns models for provider
    provider = providers[0]
    provider_models = px.models.list_working_provider_models(provider)
    assert len(provider_models) > 0
    assert all(m.provider == provider for m in provider_models)

  def test_list_working_models_with_model_size_filter(self):
    # Test model_size filter - only needs to test filtered results
    small_models = px.models.list_working_models(model_size='small')
    assert len(small_models) > 0

  def test_list_working_provider_models_with_model_size_filter(self):
    providers = px.models.list_working_providers()
    if len(providers) > 0:
      provider = providers[0]
      small_models = px.models.list_working_provider_models(
          provider, model_size='small')
      # Just verify it returns a list (may be empty for some providers)
      assert isinstance(small_models, list)


class TestQueryCache:
  """Test query caching behavior."""

  def test_cache_hit_on_second_call(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)
    px.set_model(('mock_provider', 'mock_model'))

    response1 = px.generate_text('hello', extensive_return=True)
    response2 = px.generate_text('hello', extensive_return=True)

    assert response1.response_source == types.ResponseSource.PROVIDER
    assert response2.response_source == types.ResponseSource.CACHE

  def test_use_cache_false_skips_cache(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)
    px.set_model(('mock_provider', 'mock_model'))

    response1 = px.generate_text('hello', extensive_return=True)
    response2 = px.generate_text('hello', use_cache=False, extensive_return=True)

    assert response1.response_source == types.ResponseSource.PROVIDER
    assert response2.response_source == types.ResponseSource.PROVIDER

  def test_unique_response_limit_controls_cache(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(
        cache_path=cache_path,
        unique_response_limit=2))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)
    px.set_model(('mock_provider', 'mock_model'))

    r1 = px.generate_text('hello', extensive_return=True)
    r2 = px.generate_text('hello', extensive_return=True)
    r3 = px.generate_text('hello', extensive_return=True)

    assert r1.response_source == types.ResponseSource.PROVIDER
    assert r2.response_source == types.ResponseSource.PROVIDER
    assert r3.response_source == types.ResponseSource.CACHE

  def test_use_cache_true_without_cache_raises_error(self):
    with pytest.raises(ValueError, match='use_cache is True but query cache is not working'):
      px.generate_text('hello', use_cache=True)

  def test_retry_if_error_cached_false_returns_cached_error(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)

    r1 = px.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True,
        extensive_return=True)
    r2 = px.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True,
        extensive_return=True)

    assert r1.response_source == types.ResponseSource.PROVIDER
    assert r1.response_record.error == 'Temp Error'
    assert r2.response_source == types.ResponseSource.CACHE
    assert r2.response_record.error == 'Temp Error'

  def test_retry_if_error_cached_true_retries_from_provider(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(cache_options=types.CacheOptions(
        cache_path=cache_path,
        retry_if_error_cached=True))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)

    r1 = px.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True,
        extensive_return=True)
    r2 = px.generate_text(
        'hello',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        suppress_provider_errors=True,
        extensive_return=True)

    assert r1.response_source == types.ResponseSource.PROVIDER
    assert r1.response_record.error == 'Temp Error'
    assert r2.response_source == types.ResponseSource.PROVIDER
    assert r2.response_record.error == 'Temp Error'


class TestGetCurrentOptions:
  """Test px.get_current_options() - configuration inspection."""

  def test_get_current_options_returns_run_options(self):
    options = px.get_current_options()
    assert isinstance(options, types.RunOptions)

  def test_get_current_options_reflects_connect_settings(self):
    cache_path, _ = _get_temp_dir('cache')
    px.connect(
        cache_options=types.CacheOptions(cache_path=cache_path),
        allow_multiprocessing=False,
        model_test_timeout=30)

    options = px.get_current_options()

    assert options.cache_options.cache_path == cache_path
    assert options.allow_multiprocessing == False
    assert options.model_test_timeout == 30

  def test_get_current_options_json_format(self):
    options = px.get_current_options(json=True)
    assert isinstance(options, dict)
    assert 'run_type' in options
    assert 'allow_multiprocessing' in options


class TestResetState:
  """Test px.reset_state() - state management."""

  def test_reset_state_clears_default_client(self):
    px.set_model(('mock_provider', 'mock_model'))
    px.reset_state()

    # After reset, a new client should be created
    assert px.get_default_proxai_client() is not None

  def test_reset_state_allows_fresh_connect(self):
    cache_path1, _ = _get_temp_dir('cache1')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path1))
    options1 = px.get_current_options()

    px.reset_state()

    cache_path2, _ = _get_temp_dir('cache2')
    px.connect(cache_options=types.CacheOptions(cache_path=cache_path2))
    px.get_default_proxai_client()._available_models_instance.run_type = (
        types.RunType.TEST)
    options2 = px.get_current_options()

    assert options1.cache_options.cache_path == cache_path1
    assert options2.cache_options.cache_path == cache_path2

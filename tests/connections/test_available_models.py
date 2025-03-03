import os
from typing import Dict, Optional
import proxai.types as types
from proxai import proxai
import pytest
import tempfile
import proxai.connections.available_models as available_models
import proxai.caching.model_cache as model_cache
import proxai.connectors.model_registry as model_registry
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_configs as model_configs


class TestAvailableModels:
  cache_dir: Optional[
      tempfile.TemporaryDirectory] = None
  initialized_model_connectors: Optional[
      Dict[types.ProviderModelType, model_connector.ProviderModelConnector]] = None
  model_cache_manager: Optional[
      model_cache.ModelCacheManager] = None

  @pytest.fixture(autouse=True)
  def setup_test(self, monkeypatch):
    monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
    for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
      for api_key in api_key_list:
        monkeypatch.delenv(api_key, raising=False)
    yield

  def _init_test_variables(self):
    if self.cache_dir is None:
      self.cache_dir = tempfile.TemporaryDirectory()
    if self.initialized_model_connectors is None:
      self.initialized_model_connectors = {}

  def _init_model_connector(self, provider_model: types.ProviderModelType):
    if provider_model in self.initialized_model_connectors:
      return self.initialized_model_connectors[provider_model]
    connector = model_registry.get_model_connector(provider_model)
    self.initialized_model_connectors[provider_model] = connector(
        run_type=types.RunType.TEST)
    return self.initialized_model_connectors[provider_model]

  def _get_initialized_model_connectors(self):
    return self.initialized_model_connectors

  def _get_available_models(
        self,
        allow_multiprocessing: bool = True):
    self._init_test_variables()
    self.model_cache_manager = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    available_models_manager = available_models.AvailableModels(
        run_type=types.RunType.TEST,
        allow_multiprocessing=allow_multiprocessing,
        model_cache_manager=self.model_cache_manager,
        get_initialized_model_connectors=self._get_initialized_model_connectors,
        init_model_connector=self._init_model_connector,
    )
    return available_models_manager

  def _save_temp_cache_state(self):
    self._init_test_variables()
    save_cache = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    data = types.ModelStatus()
    data.working_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4'])
    data.failed_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview'])
    data.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
            response_record=types.QueryResponseRecord(response='response1')))
    data.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS[
                    'openai']['gpt-4-turbo-preview']),
            response_record=types.QueryResponseRecord(error='error1')))
    save_cache.update(
        model_status=data, call_type=types.CallType.GENERATE_TEXT)

  def test_filter_by_key(self):
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = [
        'openai',
        'claude']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)
    assert models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview'],
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['claude']['claude-3-haiku'],
        model_configs.ALL_MODELS['claude']['claude-3-opus'],
        model_configs.ALL_MODELS['claude']['claude-3-sonnet']])
    assert models.provider_queries == []  # No queries should be filtered out since no queries exist yet

  def test_filter_by_cache(self):
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()
    available_models_manager._providers_with_key = ['openai']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)
    available_models_manager._filter_by_cache(
        models, call_type=types.CallType.GENERATE_TEXT)
    assert models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo']])
    assert models.working_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4']])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert len(models.provider_queries) == 2  # Should contain both the success and error queries from cache
    assert models.provider_queries[0].response_record.response == 'response1'
    assert models.provider_queries[1].response_record.error == 'error1'

    # Verify provider queries are properly maintained
    assert len(models.provider_queries) == 2
    assert all(
        query.query_record.provider_model in models.working_models.union(
            models.failed_models)
        for query in models.provider_queries)

  def test_filter_largest_models(self):
    available_models_manager = self._get_available_models()
    models = types.ModelStatus()

    # Add some models to working and unprocessed sets
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview'])
    models.working_models.add(
        model_configs.ALL_MODELS['claude']['claude-3-haiku'])
    models.working_models.add(
        model_configs.ALL_MODELS['claude']['claude-3-opus'])

    # Add a provider query for a model that will be filtered out
    models.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS['claude']['claude-3-haiku']),
            response_record=types.QueryResponseRecord(response='response1')))

    # Add a provider query for a model that will remain
    models.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS['claude']['claude-3-opus']),
            response_record=types.QueryResponseRecord(response='response2')))

    available_models_manager._filter_largest_models(models)

    # Check that only largest models remain in unprocessed and working sets
    assert models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert models.working_models == set([
        model_configs.ALL_MODELS['claude']['claude-3-opus']])

    # Check that filtered models were moved to filtered_models set
    assert models.filtered_models == set([
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['claude']['claude-3-haiku']])

    # Check that provider queries for filtered models were removed
    assert len(models.provider_queries) == 1
    assert models.provider_queries[0].query_record.provider_model == (
        model_configs.ALL_MODELS['claude']['claude-3-opus'])

  @pytest.mark.parametrize('allow_multiprocessing', [True, False])
  def test_test_models(self, allow_multiprocessing):
    available_models_manager = self._get_available_models(
        allow_multiprocessing=allow_multiprocessing)
    available_models_manager._providers_with_key = [
        'openai',
        'mock_failing_provider']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_key(models)

    available_models_manager._test_models(
        models, call_type=types.CallType.GENERATE_TEXT)

    assert models.unprocessed_models == set()
    assert models.working_models == set([
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model']])

    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    loaded_data = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
    assert loaded_data.working_models == set([
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert loaded_data.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model']])

  def test_get_all_models(self, monkeypatch):
    # Set only OpenAI key
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    models = available_models_manager.get_all_models()
    assert models == [
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']]

  def test_get_all_models_filters(self, monkeypatch):
    # Set only OpenAI key
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['mock_provider'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['mock_failing_provider'][0], 'test_api_key')

    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()

    # Check that the failed model was filtered out
    models = available_models_manager.get_all_models()
    assert models == [
        model_configs.ALL_MODELS['mock_provider']['mock_model'],
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4']]

    # Check cache memory values
    models = available_models_manager._model_cache_manager.get(
        call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == set([
        model_configs.ALL_MODELS['mock_provider']['mock_model'],
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4']])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert len(models.provider_queries) >= 4  # Should include original cached queries plus new test results

    # Check cache file values
    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    models = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == set([
        model_configs.ALL_MODELS['mock_provider']['mock_model'],
        model_configs.ALL_MODELS['openai']['gpt-3.5-turbo'],
        model_configs.ALL_MODELS['openai']['gpt-4']])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model'],
        model_configs.ALL_MODELS['openai']['gpt-4-turbo-preview']])
    assert len(models.provider_queries) >= 4  # Should match memory cache queries

  def test_update_provider_queries(self):
    available_models_manager = self._get_available_models()
    models = types.ModelStatus()

    # Add some test queries
    models.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
            response_record=types.QueryResponseRecord(response='response1')))
    models.provider_queries.append(
        types.LoggingRecord(
            query_record=types.QueryRecord(
                provider_model=model_configs.ALL_MODELS['claude']['claude-3-haiku']),
            response_record=types.QueryResponseRecord(response='response2')))

    # Add models to working set
    models.working_models.add(
        model_configs.ALL_MODELS['claude']['claude-3-haiku'])
    # Add models to filtered set
    models.filtered_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4'])

    # Test that queries for filtered models are removed
    available_models_manager._update_provider_queries(models)
    assert len(models.provider_queries) == 1
    assert models.provider_queries[0].query_record.provider_model == (
        model_configs.ALL_MODELS['claude']['claude-3-haiku'])

  def test_get_providers_without_cache(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    providers = available_models_manager.get_providers()
    assert providers == ['claude', 'openai']

  def test_get_providers_with_cache(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], 'test_api_key')
    available_models_manager = self._get_available_models()

    # First call should test all models since get_all_models hasn't been called
    # yet
    providers = available_models_manager.get_providers()
    assert set(providers) == set(['openai', 'claude'])

    # Second call should use existing results since get_all_models was already
    # called
    monkeypatch.delenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], raising=False)
    providers = available_models_manager.get_providers()
    assert set(providers) == set(['openai', 'claude'])

    # Clear cache and verify all providers are tested again
    providers = available_models_manager.get_providers(clear_model_cache=True)
    assert set(providers) == set(['openai'])

  def test_get_providers_invalid_call_type(self):
    available_models_manager = self._get_available_models()
    with pytest.raises(ValueError, match='Call type not supported:'):
      available_models_manager.get_providers(call_type='invalid_type')

  def test_get_providers_verbose(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    providers = available_models_manager.get_providers(verbose=True)
    assert set(providers) == set(['openai'])

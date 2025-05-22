import os
import copy
from typing import Dict, Optional, List
import proxai.types as types
from proxai import proxai
import pytest
import tempfile
import proxai.connections.available_models as available_models
import proxai.caching.model_cache as model_cache
import proxai.connectors.model_registry as model_registry
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_configs as model_configs
import proxai.connections.proxdash as proxdash
import time


@pytest.fixture(autouse=True)
def setup_test(monkeypatch):
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  yield


def get_models_set(providers: List[str]):
  models = set()
  for provider in providers:
    models.update(model_configs.GENERATE_TEXT_MODELS[provider].values())
  return models


class TestAvailableModels:
  cache_dir: Optional[
      tempfile.TemporaryDirectory] = None
  initialized_model_connectors: Optional[
      Dict[
          types.ProviderModelType,
          model_connector.ProviderModelConnector]] = None
  model_cache_manager: Optional[
      model_cache.ModelCacheManager] = None

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
        logging_options=types.LoggingOptions(),
        proxdash_connection=proxdash.ProxDashConnection(
            logging_options=types.LoggingOptions(),
            proxdash_options=types.ProxDashOptions(
                disable_proxdash=True)),
        run_type=types.RunType.TEST)
    return self.initialized_model_connectors[provider_model]

  def _get_model_connector(self, provider_model: types.ProviderModelType):
    return self._init_model_connector(provider_model)

  def _get_initialized_model_connectors(self):
    return self.initialized_model_connectors

  def _get_available_models(
        self,
        allow_multiprocessing: bool = False,
        set_model_cache_manager: bool = True):
    self._init_test_variables()
    if set_model_cache_manager:
      self.model_cache_manager = model_cache.ModelCacheManager(
          cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    available_models_manager = available_models.AvailableModels(
        run_type=types.RunType.TEST,
        get_model_connector=self._get_model_connector,
        allow_multiprocessing=allow_multiprocessing,
        model_cache_manager=(
            self.model_cache_manager if set_model_cache_manager else None),
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
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini'])
    data.provider_queries[
        model_configs.ALL_MODELS['openai']['gpt-4']
    ] = types.LoggingRecord(
        query_record=types.QueryRecord(
            provider_model=model_configs.ALL_MODELS['openai']['gpt-4']),
        response_record=types.QueryResponseRecord(response='response1'))
    data.provider_queries[
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini']
    ] = types.LoggingRecord(
        query_record=types.QueryRecord(
            provider_model=model_configs.ALL_MODELS[
                'openai']['gpt-4.1-mini']),
        response_record=types.QueryResponseRecord(error='error1'))
    save_cache.update(
        model_status_updates=data, call_type=types.CallType.GENERATE_TEXT)

  def test_filter_by_key(self):
    available_models_manager = self._get_available_models()
    available_models_manager.providers_with_key = [
        'openai',
        'claude']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_api_key(models)
    assert models.unprocessed_models == get_models_set(['openai', 'claude'])
    assert models.provider_queries == {}  # No queries should be filtered out since no queries exist yet

  def test_filter_by_cache(self):
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()
    available_models_manager.providers_with_key = ['openai']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_api_key(models)
    available_models_manager._filter_by_cache(
        models, call_type=types.CallType.GENERATE_TEXT)
    assert models.unprocessed_models == (
        get_models_set(['openai'])
        - set([
            model_configs.ALL_MODELS['openai']['gpt-4'],
            model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))
    assert models.working_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4']])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini']])
    assert len(models.provider_queries) == 2  # Should contain both the success and error queries from cache
    assert models.provider_queries[
        model_configs.ALL_MODELS['openai']['gpt-4']
    ].response_record.response == 'response1'
    assert models.provider_queries[
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini']
    ].response_record.error == 'error1'

    # Verify provider queries are properly maintained
    assert len(models.provider_queries) == 2
    assert all(
        provider_model in models.working_models.union(models.failed_models)
        for provider_model in models.provider_queries)

  def test_filter_by_model_size(self):
    available_models_manager = self._get_available_models()
    models = types.ModelStatus()

    # Add models from small, medium, large, largest model sizes
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4o-mini'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['openai']['o1-mini'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['openai']['gpt-4.1'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['claude']['haiku-3'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['claude']['haiku-3.5'])
    models.unprocessed_models.add(
        model_configs.ALL_MODELS['claude']['opus-4'])

    test_models = copy.deepcopy(models)
    available_models_manager._filter_by_model_size(
        test_models,
        model_size=types.ModelSizeType.SMALL)
    assert test_models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4o-mini'],
        model_configs.ALL_MODELS['claude']['haiku-3'],
    ])

    test_models = copy.deepcopy(models)
    available_models_manager._filter_by_model_size(
        test_models,
        model_size=types.ModelSizeType.MEDIUM)
    assert test_models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['o1-mini'],
        model_configs.ALL_MODELS['claude']['haiku-3.5'],
    ])

    test_models = copy.deepcopy(models)
    available_models_manager._filter_by_model_size(
        test_models,
        model_size=types.ModelSizeType.LARGE)
    assert test_models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4.1'],
        model_configs.ALL_MODELS['claude']['opus-4'],
    ])

    test_models = copy.deepcopy(models)
    available_models_manager._filter_by_model_size(
        test_models,
        model_size=types.ModelSizeType.LARGEST)
    assert test_models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4.1'],
        model_configs.ALL_MODELS['claude']['opus-4'],
    ])

    test_models = copy.deepcopy(models)
    available_models_manager._filter_by_model_size(
        test_models,
        model_size='small')
    assert test_models.unprocessed_models == set([
        model_configs.ALL_MODELS['openai']['gpt-4o-mini'],
        model_configs.ALL_MODELS['claude']['haiku-3'],
    ])

  @pytest.mark.parametrize('allow_multiprocessing', [True, False])
  def test_test_models(self, allow_multiprocessing):
    available_models_manager = self._get_available_models(
        allow_multiprocessing=allow_multiprocessing)
    available_models_manager.providers_with_key = [
        'openai',
        'mock_failing_provider']
    models = types.ModelStatus()
    available_models_manager._get_all_models(
        models, call_type=types.CallType.GENERATE_TEXT)
    available_models_manager._filter_by_provider_api_key(models)

    available_models_manager._test_models(
        models, call_type=types.CallType.GENERATE_TEXT)

    assert models.unprocessed_models == set()
    assert models.working_models == get_models_set(['openai'])
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model']])

  def test_get_all_models(self, monkeypatch):
    # Set only OpenAI key
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    models = available_models_manager.list_models()
    assert models == sorted(list(get_models_set(['openai'])))

  def test_get_all_models_filters(self, monkeypatch):
    # Set only OpenAI key
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['mock_provider'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['mock_failing_provider'][0],
        'test_api_key')

    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()

    # Check that the failed model was filtered out
    models = available_models_manager.list_models()
    assert models == sorted(list(
        get_models_set(['openai', 'mock_provider'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']])))

    # Check cache memory values
    models = available_models_manager._model_cache_manager.get(
        call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == (
        get_models_set(['openai', 'mock_provider'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model'],
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini']])
    assert len(models.provider_queries) >= 4  # Should include original cached queries plus new test results

    # Check cache file values
    load_cache = model_cache.ModelCacheManager(
        cache_options=types.CacheOptions(cache_path=self.cache_dir.name))
    models = load_cache.get(call_type=types.CallType.GENERATE_TEXT)
    assert models.working_models == (
        get_models_set(['openai', 'mock_provider'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))
    assert models.failed_models == set([
        model_configs.ALL_MODELS['mock_failing_provider']['mock_failing_model'],
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini']])
    assert len(models.provider_queries) >= 4  # Should match memory cache queries

  def test_get_providers_without_cache(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], 'test_api_key')
    available_models_manager = self._get_available_models(
        set_model_cache_manager=False)
    providers = available_models_manager.list_providers()
    assert providers == ['claude', 'openai']

  def test_get_providers_with_cache(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], 'test_api_key')
    available_models_manager = self._get_available_models()

    providers = available_models_manager.list_providers()
    assert set(providers) == set(['openai', 'claude'])

    monkeypatch.delenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], raising=False)
    providers = available_models_manager.list_providers()
    assert set(providers) == set(['openai'])

  def test_get_providers_invalid_call_type(self):
    available_models_manager = self._get_available_models()
    with pytest.raises(ValueError, match='Call type not supported:'):
      available_models_manager.list_providers(call_type='invalid_type')

  def test_get_providers_verbose(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    providers = available_models_manager.list_providers(verbose=True)
    assert set(providers) == set(['openai'])

  def test_get_provider_models_without_cache(self, monkeypatch):
    # Set only OpenAI key
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models(
        set_model_cache_manager=False)

    # Test provider with key
    models = available_models_manager.list_provider_models('openai')
    assert set(models) == set(
        model_configs.GENERATE_TEXT_MODELS['openai'].values())

    # Test provider without key
    with pytest.raises(
        ValueError,
        match='Provider key not found in environment variables for claude.\n'
        'Required keys'):
      available_models_manager.list_provider_models('claude')

  def test_get_provider_models_with_cache(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], 'test_api_key')
    available_models_manager = self._get_available_models()

    models = available_models_manager.list_provider_models('openai')
    # 'gpt-4.1-mini' saved as failed model, so it should not be included
    assert set(models) == (
        get_models_set(['openai'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))

    models = available_models_manager.list_provider_models('claude')
    assert set(models) == set(
        model_configs.GENERATE_TEXT_MODELS['claude'].values())

    monkeypatch.delenv(
        model_configs.PROVIDER_KEY_MAP['claude'][0], raising=False)
    models = available_models_manager.list_provider_models('openai')
    # 'gpt-4.1-mini' saved as failed model, so it should not be included
    assert set(models) == (
        get_models_set(['openai'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))
    models = available_models_manager.list_provider_models('claude')
    assert set(models) == set()

  def test_get_provider_models_invalid_call_type(self):
    available_models_manager = self._get_available_models()
    with pytest.raises(ValueError, match='Call type not supported:'):
      available_models_manager.list_provider_models(
          'openai', call_type='invalid_type')

  def test_get_provider_models_verbose(self, monkeypatch):
    self._save_temp_cache_state()
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()
    models = available_models_manager.list_provider_models(
        'openai', verbose=True)
    assert set(models) == (
        get_models_set(['openai'])
        - set([model_configs.ALL_MODELS['openai']['gpt-4.1-mini']]))

  def test_get_provider_model_without_cache_manager(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models(
        set_model_cache_manager=False)

    # Test successful case
    provider_model = available_models_manager.get_model(
        'openai', 'gpt-4')
    assert provider_model == model_configs.ALL_MODELS['openai']['gpt-4']

    # Test provider without key
    with pytest.raises(
        ValueError,
        match='Provider key not found in environment variables for claude.'):
      available_models_manager.get_model('claude', 'haiku-3.5')

    # Test invalid provider
    with pytest.raises(
        ValueError,
        match='Provider not found in model_configs: invalid_provider'):
      available_models_manager.get_model('invalid_provider', 'model')

    # Test invalid model
    with pytest.raises(
        ValueError,
        match='Model not found in openai models: invalid_model'):
      available_models_manager.get_model('openai', 'invalid_model')

  def test_get_provider_model_with_cache_manager(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()

    # Test successful case with cached model
    provider_model = available_models_manager.get_model(
        'openai', 'gpt-4')
    assert provider_model == model_configs.ALL_MODELS['openai']['gpt-4']

    # Test model not in working models
    with pytest.raises(
        ValueError,
        match='Provider model not found in working models'):
      available_models_manager.get_model(
          'openai', 'gpt-4.1-mini')

  def test_get_provider_model_clear_cache(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    available_models_manager = self._get_available_models()

    provider_model = available_models_manager.get_model(
        'openai', 'gpt-4')
    assert provider_model == model_configs.ALL_MODELS['openai']['gpt-4']

    monkeypatch.delenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], raising=False)
    with pytest.raises(
        ValueError,
        match='Provider model not found in working models'):
      available_models_manager.get_model(
          'openai', 'gpt-4', clear_model_cache=True)

  def test_get_provider_model_with_ignore_model_status(self, monkeypatch):
    monkeypatch.setenv(
        model_configs.PROVIDER_KEY_MAP['openai'][0], 'test_api_key')
    self._save_temp_cache_state()
    available_models_manager = self._get_available_models()

    # Check that the model is returned even if it is in the failed models
    provider_model = available_models_manager.get_model(
        'openai', 'gpt-4.1-mini', allow_non_working_model=True)
    assert provider_model == model_configs.ALL_MODELS['openai']['gpt-4.1-mini']

    assert (
        model_configs.ALL_MODELS['openai']['gpt-4.1-mini'] in
        available_models_manager.list_models(return_all=True).failed_models)

  def test_get_provider_model_invalid_call_type(self):
    available_models_manager = self._get_available_models()
    with pytest.raises(ValueError, match='Call type not supported:'):
      available_models_manager.get_model(
          'openai', 'gpt-4', call_type='invalid_type')


class TestAvailableModelsState:
  # TODO: Add tests for AvailableModels state controlled properties
  pass

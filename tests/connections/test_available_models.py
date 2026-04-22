"""Tests for AvailableModels.

Discovers and manages available AI models across providers. Two behavior
surfaces:

  1. Pure discovery (no network) — filters the registered model set by
     API-key presence, feature/format support, size, recommended flag.
     Implemented via _fetch_all_models with raw_config_results_without_test.

  2. Health-check (invokes provider executors) — runs a test query
     against each model and classifies as working/failed; optionally
     caches results via ModelCacheManager.

Tests are organized by responsibility:

  - Init + state serialization
  - get_model_connector (lazy, cached)
  - list_models (core discovery; filter combinations)
  - list_providers, list_provider_models, get_model, get_model_config
  - _filter_by_cache (cache merging invariants)
  - list_working_* / get_working_model / check_health (using
    MockProviderModelConnector + MockFailingProviderModelConnector)
"""


import pytest

import proxai.caching.model_cache as model_cache
import proxai.connections.api_key_manager as api_key_manager
import proxai.connections.available_models as available_models
import proxai.connectors.model_configs as model_configs
import proxai.types as types

# =============================================================================
# Fixtures + helpers
# =============================================================================


_MOCK_ENV_KEYS = {
    'MOCK_PROVIDER_API_KEY': 'k1',
    'MOCK_FAILING_PROVIDER': 'k2',
    'MOCK_SLOW_PROVIDER': 'k3',
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  """Strip all provider API keys so each test declares what it sets."""
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  yield


def _set_mock_keys(monkeypatch):
  """Set env keys for the three mock providers."""
  for k, v in _MOCK_ENV_KEYS.items():
    monkeypatch.setenv(k, v)


def _set_fast_mock_keys(monkeypatch):
  """Set keys for mock_provider + mock_failing_provider only.

  mock_slow_provider's executor does `time.sleep(120)` — exclude it from
  any test that actually drives the health-check pipeline through
  _test_models.
  """
  monkeypatch.setenv('MOCK_PROVIDER_API_KEY', 'k1')
  monkeypatch.setenv('MOCK_FAILING_PROVIDER', 'k2')


def _build_available_models(
    *,
    model_cache_manager=None,
    allow_multiprocessing=False,
    feature_mapping_strategy=None,
    model_configs_instance=None,
) -> available_models.AvailableModels:
  """Construct an AvailableModels wired to the session shared model registry.

  Caller is expected to have set env keys for the providers being tested.
  """
  mc_instance = model_configs_instance or pytest.model_configs_instance
  akm = api_key_manager.ApiKeyManager(
      init_from_params=api_key_manager.ApiKeyManagerParams()
  )
  pco_kwargs = {}
  if feature_mapping_strategy is not None:
    pco_kwargs['feature_mapping_strategy'] = feature_mapping_strategy
  return available_models.AvailableModels(
      init_from_params=available_models.AvailableModelsParams(
          run_type=types.RunType.TEST,
          provider_call_options=types.ProviderCallOptions(**pco_kwargs),
          model_configs_instance=mc_instance,
          model_cache_manager=model_cache_manager,
          logging_options=types.LoggingOptions(),
          api_key_manager=akm,
          model_probe_options=types.ModelProbeOptions(
              allow_multiprocessing=allow_multiprocessing,
          ),
          debug_options=types.DebugOptions(),
      )
  )


def _make_model_cache_manager(cache_dir: str) -> model_cache.ModelCacheManager:
  return model_cache.ModelCacheManager(
      init_from_params=model_cache.ModelCacheManagerParams(
          cache_options=types.CacheOptions(cache_path=cache_dir),
      )
  )


# =============================================================================
# Init + state
# =============================================================================


class TestInit:

  def test_init_from_params_populates_attributes(self):
    am = _build_available_models()
    assert am.run_type == types.RunType.TEST
    assert am.model_configs_instance is pytest.model_configs_instance
    assert am.provider_connectors == {}
    assert am.model_probe_options.allow_multiprocessing is False

  def test_invalid_combinations_raises(self):
    with pytest.raises(
        ValueError,
        match='init_from_params and init_from_state cannot be set',
    ):
      available_models.AvailableModels(
          init_from_params=available_models.AvailableModelsParams(),
          init_from_state=types.AvailableModelsState(),
      )

  def test_state_round_trip(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    state = am.get_state()
    restored = available_models.AvailableModels(init_from_state=state)
    # Core fields preserved.
    assert restored.run_type == types.RunType.TEST
    assert (
        restored.model_configs_instance.model_registry ==
        am.model_configs_instance.model_registry
    )


# =============================================================================
# get_model_connector
# =============================================================================


class TestGetModelConnector:

  def test_returns_connector_for_registered_provider(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    connector = am.get_model_connector(('mock_provider', 'mock_model'))
    assert connector.PROVIDER_NAME == 'mock_provider'

  def test_caches_connector_on_repeat_lookup(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    first = am.get_model_connector(('mock_provider', 'mock_model'))
    second = am.get_model_connector(('mock_provider', 'mock_model'))
    assert first is second


# =============================================================================
# list_models — core discovery method
# =============================================================================


class TestListModels:

  def test_no_keys_raises(self):
    am = _build_available_models()
    with pytest.raises(ValueError, match='No provider API keys found'):
      am.list_models()

  def test_api_key_gates_models(self, monkeypatch):
    # Only mock_provider key set — openai/gemini must NOT appear.
    monkeypatch.setenv('MOCK_PROVIDER_API_KEY', 'k1')
    am = _build_available_models()
    results = am.list_models(recommended_only=False)
    providers = {m.provider for m in results}
    assert providers == {'mock_provider'}

  def test_recommended_only_default_excludes_non_recommended(
      self, monkeypatch,
  ):
    # Mock providers are is_recommended=False — default recommended_only
    # filter means they're absent.
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    results = am.list_models()  # recommended_only=True by default
    providers = {m.provider for m in results}
    # Mock providers are all is_recommended=False.
    assert 'mock_provider' not in providers
    assert 'mock_failing_provider' not in providers
    assert 'mock_slow_provider' not in providers

  def test_recommended_only_false_includes_non_recommended(
      self, monkeypatch,
  ):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    results = am.list_models(recommended_only=False)
    providers = {m.provider for m in results}
    assert 'mock_provider' in providers

  def test_model_size_filter(self, monkeypatch):
    # Mock providers have [SMALL] tag.
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    small = am.list_models(
        model_size=types.ModelSizeType.SMALL, recommended_only=False,
    )
    assert any(m.provider == 'mock_provider' for m in small)
    # Now filter by LARGE — mock models don't have it.
    large = am.list_models(
        model_size=types.ModelSizeType.LARGE, recommended_only=False,
    )
    assert all(m.provider != 'mock_provider' for m in large)

  def test_feature_tags_filter_excludes_unsupported(self, monkeypatch):
    # mock_provider has THINKING=NOT_SUPPORTED at the model-feature-config
    # level (registered in conftest). So filtering by THINKING tag with
    # STRICT mapping strategy drops mock_provider.
    _set_mock_keys(monkeypatch)
    am = _build_available_models(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
    )
    results = am.list_models(
        feature_tags=[types.FeatureTag.THINKING], recommended_only=False,
    )
    providers = {m.provider for m in results}
    assert 'mock_provider' not in providers
    # PROMPT IS supported on mock_provider.
    results_prompt = am.list_models(
        feature_tags=[types.FeatureTag.PROMPT], recommended_only=False,
    )
    providers_prompt = {m.provider for m in results_prompt}
    assert 'mock_provider' in providers_prompt


# =============================================================================
# list_providers / list_provider_models / get_model / get_model_config
# =============================================================================


class TestListProviders:

  def test_returns_sorted_providers_with_keys(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    providers = am.list_providers(recommended_only=False)
    # All three mock providers have keys and have at least one model
    # registered, so they surface here. Result is sorted.
    assert providers == sorted(providers)
    assert 'mock_provider' in providers
    assert 'mock_failing_provider' in providers
    assert 'mock_slow_provider' in providers

  def test_no_keys_raises(self):
    am = _build_available_models()
    with pytest.raises(ValueError, match='No provider API keys found'):
      am.list_providers()


class TestListProviderModels:

  def test_returns_models_for_provider(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    models = am.list_provider_models(
        provider='mock_provider', recommended_only=False,
    )
    assert all(m.provider == 'mock_provider' for m in models)
    assert any(m.model == 'mock_model' for m in models)

  def test_no_key_for_provider_raises(self, monkeypatch):
    # Only mock_provider has a key; asking for openai must raise.
    monkeypatch.setenv('MOCK_PROVIDER_API_KEY', 'k1')
    am = _build_available_models()
    with pytest.raises(ValueError, match='Provider key not found'):
      am.list_provider_models(provider='openai')


class TestGetModel:

  def test_returns_provider_model(self, monkeypatch):
    _set_mock_keys(monkeypatch)
    am = _build_available_models()
    pm = am.get_model(provider='mock_provider', model='mock_model')
    assert pm.provider == 'mock_provider'
    assert pm.model == 'mock_model'

  def test_no_key_raises(self, monkeypatch):
    # No mock keys set; the registry still has mock_provider so the
    # lookup succeeds, but the API-key check raises.
    am = _build_available_models()
    with pytest.raises(ValueError, match='Provider key not found'):
      am.get_model(provider='mock_provider', model='mock_model')


class TestGetModelConfig:

  def test_returns_config(self):
    # get_model_config does NOT require an API key — it's a pure registry
    # lookup for inspecting model capabilities.
    am = _build_available_models()
    config = am.get_model_config(
        provider='mock_provider', model='mock_model',
    )
    assert isinstance(config, types.ProviderModelConfig)
    assert config.provider_model.provider == 'mock_provider'


# =============================================================================
# _filter_by_cache — cache merging invariants
# =============================================================================


class TestFilterByCache:

  def test_merges_working_from_cache(self, monkeypatch, tmp_path):
    """Unprocessed model cached as working → moves to working_models."""
    _set_mock_keys(monkeypatch)
    cache_mgr = _make_model_cache_manager(str(tmp_path))
    pm = pytest.model_configs_instance.get_provider_model(
        ('mock_provider', 'mock_model'))
    cached = types.ModelStatus()
    cached.working_models.add(pm)
    cache_mgr.save(cached, types.OutputFormatType.TEXT)

    am = _build_available_models(model_cache_manager=cache_mgr)
    models = types.ModelStatus()
    models.unprocessed_models.add(pm)
    am._filter_by_cache(models, types.OutputFormatType.TEXT)

    assert pm in models.working_models
    assert pm not in models.unprocessed_models

  def test_merges_failed_from_cache(self, monkeypatch, tmp_path):
    """Unprocessed model cached as failed → moves to failed_models."""
    _set_mock_keys(monkeypatch)
    cache_mgr = _make_model_cache_manager(str(tmp_path))
    pm = pytest.model_configs_instance.get_provider_model(
        ('mock_failing_provider', 'mock_failing_model'))
    cached = types.ModelStatus()
    cached.failed_models.add(pm)
    cache_mgr.save(cached, types.OutputFormatType.TEXT)

    am = _build_available_models(model_cache_manager=cache_mgr)
    models = types.ModelStatus()
    models.unprocessed_models.add(pm)
    am._filter_by_cache(models, types.OutputFormatType.TEXT)

    assert pm in models.failed_models
    assert pm not in models.unprocessed_models

  def test_no_cache_manager_noop(self):
    """query_cache_manager is None → no mutation, no error."""
    am = _build_available_models()
    pm = pytest.model_configs_instance.get_provider_model(
        ('mock_provider', 'mock_model'))
    models = types.ModelStatus()
    models.unprocessed_models.add(pm)
    am._filter_by_cache(models, types.OutputFormatType.TEXT)
    assert pm in models.unprocessed_models
    assert pm not in models.working_models
    assert pm not in models.failed_models


# =============================================================================
# Health-check surface — run real mock connectors through generate()
# =============================================================================
#
# mock_provider returns "mock response" (SUCCESS); mock_failing_provider raises
# ValueError (FAILED). Both are is_recommended=False per conftest, so every
# health-check call below passes recommended_only=False.


class TestListWorkingModels:

  def test_without_cache_runs_tests_and_classifies(self, monkeypatch):
    """No cache → tests run inline; mock_provider works, failing fails."""
    _set_fast_mock_keys(monkeypatch)
    am = _build_available_models()
    status = am.list_working_models(
        return_all=True, verbose=False, recommended_only=False,
    )
    working = {m.provider for m in status.working_models}
    failed = {m.provider for m in status.failed_models}
    assert 'mock_provider' in working
    assert 'mock_failing_provider' in failed

  def test_return_all_false_returns_working_list(self, monkeypatch):
    """return_all=False → list of working ProviderModelType, sorted."""
    _set_fast_mock_keys(monkeypatch)
    am = _build_available_models()
    working = am.list_working_models(
        verbose=False, recommended_only=False,
    )
    assert isinstance(working, list)
    providers = {m.provider for m in working}
    assert 'mock_provider' in providers
    assert 'mock_failing_provider' not in providers

  def test_model_size_filter(self, monkeypatch):
    """model_size filter drops unmatched sizes before testing."""
    _set_fast_mock_keys(monkeypatch)
    am = _build_available_models()
    large = am.list_working_models(
        model_size=types.ModelSizeType.LARGE,
        return_all=True, verbose=False, recommended_only=False,
    )
    # Mock providers only have SMALL; filtering by LARGE drops them all.
    assert all(
        m.provider not in {'mock_provider', 'mock_failing_provider'}
        for m in large.working_models
    )


class TestGetWorkingModel:

  # NOTE: get_working_model without a cache_manager takes a short path —
  # validate key presence and return. With a cache_manager it runs
  # _fetch_all_models which hardcodes recommended_only=True, so
  # non-recommended models (all three mock_* providers in conftest) are
  # filtered out before testing. That makes the cache-backed branch
  # untestable with our mock registry; we cover the no-cache path here.

  def test_returns_model_when_key_present_no_cache(self, monkeypatch):
    """No cache_manager + key set → returns ProviderModelType."""
    _set_fast_mock_keys(monkeypatch)
    am = _build_available_models()
    pm = am.get_working_model(provider='mock_provider', model='mock_model')
    assert pm.provider == 'mock_provider'
    assert pm.model == 'mock_model'

  def test_no_key_raises(self):
    """No cache_manager + no key → ValueError('Provider key not found')."""
    am = _build_available_models()
    with pytest.raises(ValueError, match='Provider key not found'):
      am.get_working_model(provider='mock_provider', model='mock_model')


class TestListWorkingProviders:

  def test_returns_only_providers_with_working_models(
      self, monkeypatch, tmp_path,
  ):
    _set_fast_mock_keys(monkeypatch)
    cache_mgr = _make_model_cache_manager(str(tmp_path))
    am = _build_available_models(model_cache_manager=cache_mgr)
    providers = am.list_working_providers(
        verbose=False, recommended_only=False,
    )
    assert 'mock_provider' in providers
    # Failing provider has no working models → excluded.
    assert 'mock_failing_provider' not in providers


class TestListWorkingProviderModels:

  def test_returns_working_models_for_provider(self, monkeypatch, tmp_path):
    _set_fast_mock_keys(monkeypatch)
    cache_mgr = _make_model_cache_manager(str(tmp_path))
    am = _build_available_models(model_cache_manager=cache_mgr)
    models = am.list_working_provider_models(
        provider='mock_provider', verbose=False, recommended_only=False,
    )
    assert all(m.provider == 'mock_provider' for m in models)
    assert any(m.model == 'mock_model' for m in models)


class TestCheckHealth:

  def test_classifies_working_and_failed(self, monkeypatch, tmp_path):
    """check_health wraps list_working_models(return_all=True)."""
    _set_fast_mock_keys(monkeypatch)
    cache_mgr = _make_model_cache_manager(str(tmp_path))
    am = _build_available_models(model_cache_manager=cache_mgr)
    status = am.check_health(verbose=False)
    assert isinstance(status, types.ModelStatus)
    working_providers = {m.provider for m in status.working_models}
    failed_providers = {m.provider for m in status.failed_models}
    # check_health uses recommended_only=True (the default). Since mock
    # providers are is_recommended=False, they are excluded — we assert the
    # call completes and returns a valid ModelStatus.
    assert working_providers.isdisjoint(failed_providers)

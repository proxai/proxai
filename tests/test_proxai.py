"""Tests for the `px.*` module facade.

Focus: thin-forwarding smoke + singleton lifecycle + user-pattern integration
tests. Deep coverage of the `ProxAIClient` surface itself lives in
`test_client.py`; this file proves that the module-level `px.foo(...)`
entry points reach the default client and that the common user flows
(connect → generate → reset) work end-to-end.

Tests are organized by responsibility:

  - TestFacadeForwarding: each public `px.*` forwards to the default client.
  - TestSingletonLifecycle: `_DEFAULT_CLIENT` idempotent / reset / replace.
  - TestModelsFilesSingletons: `px.models` and `px.files` identity.
  - TestUserPatterns: the seven load-bearing user flows:
      1. zero-config generate_text
      2. set_model → generate
      3. alias dispatch (per-format defaults)
      4. connect → cache hit
      5. client-wide suppress via provider_call_options
      6. per-call suppress via connection_options
      7. reset_state lifecycle
"""

import pytest

import proxai as px
import proxai.client as client
import proxai.connectors.model_configs as model_configs
import proxai.types as types

# =============================================================================
# Fixtures + helpers
# =============================================================================


_MOCK_KEYS = {
    'MOCK_PROVIDER_API_KEY': 'k1',
    'MOCK_FAILING_PROVIDER': 'k2',
}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
  """Strip provider keys, set fast mocks, reset the px singleton each test.

  The module-level `_DEFAULT_CLIENT` must not leak between tests — otherwise
  a `px.connect()` in test A would colour test B's default client.
  """
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  for k, v in _MOCK_KEYS.items():
    monkeypatch.setenv(k, v)
  px.reset_state()
  yield
  px.reset_state()


def _wire_default_client_for_mocks(*, with_mock_model: bool = True) -> None:
  """Prepare the default client to run against mock providers.

  - Forces run_type=TEST on AvailableModels.
  - Registers mock_provider + mock_failing_provider configs.
  - Optionally sets the TEXT default to mock_provider.
  """
  # Import the registration helper from test_client.py's approach, inline.
  px_client = px.get_default_proxai_client()
  px_client.available_models_instance.run_type = types.RunType.TEST
  _register_mock_providers(px_client.model_configs_instance)
  if with_mock_model:
    px.set_model(('mock_provider', 'mock_model'))


def _register_mock_providers(mc_instance: model_configs.ModelConfigs) -> None:
  """Register mock_provider + mock_failing_provider into mc_instance."""
  S = types.FeatureSupportType.SUPPORTED
  NS = types.FeatureSupportType.NOT_SUPPORTED
  existing = mc_instance.model_registry.provider_model_configs
  for provider, model in [
      ('mock_provider', 'mock_model'),
      ('mock_failing_provider', 'mock_failing_model'),
  ]:
    if provider in existing and model in existing[provider]:
      continue
    mc_instance.register_provider_model_config(
        types.ProviderModelConfig(
            provider_model=types.ProviderModelType(
                provider=provider, model=model,
                provider_model_identifier=model),
            pricing=types.ProviderModelPricingType(
                input_token_cost=1, output_token_cost=2),
            metadata=types.ProviderModelMetadataType(
                is_recommended=False,
                model_size_tags=[types.ModelSizeType.SMALL]),
            features=types.FeatureConfigType(
                prompt=S, messages=S, system_prompt=S,
                parameters=types.ParameterConfigType(
                    temperature=S, max_tokens=S, stop=NS, n=NS, thinking=NS),
                tools=types.ToolConfigType(web_search=NS),
                input_format=types.InputFormatConfigType(
                    text=S, image=NS, document=NS, audio=NS, video=NS,
                    json=NS, pydantic=NS),
                output_format=types.OutputFormatConfigType(
                    text=S, json=S, pydantic=S, image=NS, audio=NS,
                    video=NS),
            ),
        )
    )


def _px_connect_disable_proxdash(**kwargs) -> None:
  """px.connect() wrapper that defaults to proxdash_options=disabled."""
  kwargs.setdefault(
      'proxdash_options', types.ProxDashOptions(disable_proxdash=True),
  )
  px.connect(**kwargs)


# =============================================================================
# Facade forwarding — each px.foo(...) reaches the default client
# =============================================================================


class TestFacadeForwarding:

  def test_connect_creates_default_client(self):
    import proxai.proxai as proxai_module

    assert proxai_module._DEFAULT_CLIENT is None
    _px_connect_disable_proxdash()
    assert proxai_module._DEFAULT_CLIENT is not None
    assert isinstance(proxai_module._DEFAULT_CLIENT, client.ProxAIClient)

  def test_get_default_client_lazy_creates(self):
    """First call builds the singleton even without connect()."""
    c = px.get_default_proxai_client()
    assert isinstance(c, client.ProxAIClient)

  def test_set_model_forwards(self):
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks(with_mock_model=False)
    px.set_model(('mock_provider', 'mock_model'))
    registered = px.get_default_proxai_client().registered_models
    assert types.OutputFormatType.TEXT in registered

  def test_generate_forwards_returns_call_record(self):
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks()
    result = px.generate(prompt='hi')
    assert isinstance(result, types.CallRecord)
    assert result.result.status == types.ResultStatusType.SUCCESS

  def test_generate_text_forwards_returns_string(self):
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks()
    assert px.generate_text(prompt='hi') == 'mock response'

  def test_generate_json_forwards_returns_dict(self):
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks()
    result = px.generate_json(prompt='hi')
    assert isinstance(result, dict)
    assert result == {'name': 'John Doe', 'age': 30}

  def test_get_current_options_forwards(self):
    _px_connect_disable_proxdash(experiment_path='facade/exp')
    options = px.get_current_options()
    assert isinstance(options, types.RunOptions)
    assert options.experiment_path == 'facade/exp'


# =============================================================================
# Singleton lifecycle — reset_state / idempotent / replace
# =============================================================================


class TestSingletonLifecycle:

  def test_get_default_client_is_idempotent(self):
    """Repeated calls return the same singleton object."""
    a = px.get_default_proxai_client()
    b = px.get_default_proxai_client()
    assert a is b

  def test_reset_state_clears_singleton(self):
    import proxai.proxai as proxai_module

    _px_connect_disable_proxdash()
    assert proxai_module._DEFAULT_CLIENT is not None
    px.reset_state()
    assert proxai_module._DEFAULT_CLIENT is None

  def test_connect_replaces_existing_client(self):
    """Second connect() builds a fresh client with new config."""
    _px_connect_disable_proxdash(experiment_path='before')
    first = px.get_default_proxai_client()
    _px_connect_disable_proxdash(experiment_path='after')
    second = px.get_default_proxai_client()
    assert first is not second
    assert second.experiment_path == 'after'


# =============================================================================
# px.models / px.files — singleton identity
# =============================================================================


class TestModelsFilesSingletons:

  def test_px_models_delegates_to_default_client(self):
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks(with_mock_model=False)
    # list_models with recommended_only=False reaches the client's AvailableModels.
    results = px.models.list_models(recommended_only=False)
    providers = {m.provider for m in results}
    assert 'mock_provider' in providers

  def test_px_files_delegates_to_default_client(self):
    _px_connect_disable_proxdash()
    # is_download_supported forwards to files_manager — no network needed.
    assert px.files.is_download_supported('gemini') is False


# =============================================================================
# px.models.model_config — module-level facade for registry mutation
# =============================================================================


def _make_provider_model_config(
    provider: str,
    model: str,
    identifier: str | None = None,
) -> types.ProviderModelConfig:
  """Minimal valid ProviderModelConfig for registry-mutation tests."""
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider, model=model,
          provider_model_identifier=identifier or model),
      pricing=types.ProviderModelPricingType(
          input_token_cost=1, output_token_cost=1),
      features=types.FeatureConfigType(),
      metadata=types.ProviderModelMetadataType(),
  )


class TestModelConfigFacade:

  def test_px_model_config_bound_to_default_client(self):
    _px_connect_disable_proxdash()
    px_client = px.get_default_proxai_client()
    assert px.models.model_config._client_getter() is px_client

  def test_register_reaches_default_client(self):
    _px_connect_disable_proxdash()
    config = _make_provider_model_config('testp', 'testm')
    px.models.model_config.register_provider_model_config(config)
    default_client = px.get_default_proxai_client()
    assert default_client.model_configs_instance.get_provider_model(
        ('testp', 'testm')).model == 'testm'

  def test_unregister_all_reaches_default_client(self):
    _px_connect_disable_proxdash()
    px.models.model_config.unregister_all_models()
    registry = (
        px.get_default_proxai_client().model_configs_instance.model_registry)
    assert registry.provider_model_configs == {}
    assert registry.default_model_priority_list == []

  def test_get_default_model_priority_list_reaches_default_client(self):
    _px_connect_disable_proxdash()
    via_facade = px.models.model_config.get_default_model_priority_list()
    via_client = (
        px.get_default_proxai_client().model_configs_instance
        .get_default_model_priority_list())
    assert via_facade is via_client


# =============================================================================
# TestUserPatterns — seven load-bearing user flows.
#
# Each test mirrors a real user journey that a common tutorial / README /
# integration example exercises. A regression in any of these would surface
# the very first time a user tried the corresponding pattern.
# =============================================================================


class TestUserPatterns:

  def test_pattern_1_zero_config_generate_text(self):
    """A user calls px.generate_text() after set_model() with no connect().

    No connect() needed — the default client is built lazily. This is
    the simplest hello-world flow in the README.
    """
    _wire_default_client_for_mocks()
    result = px.generate_text(prompt='hello')
    assert result == 'mock response'

  def test_pattern_2_set_model_then_generate_propagates_default(self):
    """User sets a default model once, then generate() uses it implicitly.

    This is the most common config pattern. The test asserts the CallRecord's
    provider_model matches the set default — proving the default flows all
    the way through generate() without being passed per-call.
    """
    _wire_default_client_for_mocks(with_mock_model=False)
    px.set_model(('mock_provider', 'mock_model'))
    result = px.generate(prompt='hi')
    assert result.query.provider_model.provider == 'mock_provider'
    assert result.query.provider_model.model == 'mock_model'

  def test_pattern_3_alias_dispatch_per_format(self):
    """Per-format set_model — generate_text vs generate_json route separately.

    User configures different models for different output formats. Each alias
    must route to its format-specific default.
    """
    _wire_default_client_for_mocks(with_mock_model=False)
    px.set_model(
        generate_text=('mock_provider', 'mock_model'),
        generate_json=('mock_provider', 'mock_model'),
    )
    assert px.generate_text(prompt='hi') == 'mock response'
    assert px.generate_json(prompt='hi') == {'name': 'John Doe', 'age': 30}

  def test_pattern_4_connect_cache_hit(self, tmp_path):
    """User configures a cache via px.connect() → second call hits cache.

    The second call must return `result_source == CACHE`. This proves the
    cache wiring flows from px.connect() through to the connector layer.
    """
    _px_connect_disable_proxdash(
        cache_options=types.CacheOptions(cache_path=str(tmp_path)),
    )
    _wire_default_client_for_mocks()
    first = px.generate(prompt='cache me')
    second = px.generate(prompt='cache me')
    assert first.connection.result_source == types.ResultSource.PROVIDER
    assert second.connection.result_source == types.ResultSource.CACHE

  def test_pattern_5_suppress_errors_via_provider_call_options(self):
    """Client-wide error suppression via provider_call_options.

    User sets suppress_provider_errors=True at connect time → a failing
    provider returns a stringified error instead of raising.
    """
    _px_connect_disable_proxdash(
        provider_call_options=types.ProviderCallOptions(
            suppress_provider_errors=True,
        ),
    )
    _wire_default_client_for_mocks(with_mock_model=False)
    result = px.generate_text(
        prompt='hi',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
    )
    # With suppression, the alias returns the error string, not a raise.
    assert isinstance(result, str)
    assert 'Mock failing provider query' in result

  def test_pattern_6_suppress_errors_via_connection_options(self):
    """Per-call error suppression via connection_options overrides the default.

    User defaults to strict errors but allows one call to be forgiving.
    """
    # No client-wide suppression.
    _px_connect_disable_proxdash()
    _wire_default_client_for_mocks(with_mock_model=False)
    result = px.generate(
        prompt='hi',
        provider_model=('mock_failing_provider', 'mock_failing_model'),
        connection_options=types.ConnectionOptions(
            suppress_provider_errors=True,
        ),
    )
    assert result.result.status == types.ResultStatusType.FAILED
    assert 'Mock failing provider query' in result.result.error

  def test_pattern_7_reset_state_lifecycle(self):
    """User sets experiment, then reset_state → fresh client with no experiment.

    Covers the full lifecycle of the module singleton.
    """
    _px_connect_disable_proxdash(experiment_path='exp/one')
    assert px.get_current_options().experiment_path == 'exp/one'
    px.reset_state()
    # After reset, get_current_options builds a fresh default client.
    # The new client has no experiment_path.
    assert px.get_current_options().experiment_path is None

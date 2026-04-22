"""Tests for ProxAIClient — the low-level SDK client class.

Focus: function-level coverage of `ProxAIClient` — each public method,
every property with non-trivial setter logic, every error branch of
`generate()`. User-pattern integration tests live in `test_proxai.py`.

Tests are organized by responsibility:

  - Params + Init (construction + state round-trip + conflict errors)
  - Properties with non-trivial setters
  - Cache manager wiring (lazy init, None fallback)
  - get_default_provider_model (fallback ladder)
  - set_model (variants + error paths)
  - generate() validation (six error branches)
  - Alias smoke (generate_text / generate_json / generate_pydantic)
  - fallback_models happy path
  - get_current_options / get_state
  - ModelConnector + FileConnector (delegation smoke)
  - keep_raw_provider_response (debug invariant)

Tests use MockProviderModelConnector + MockFailingProviderModelConnector
via `run_type=TEST` — no network.
"""

import pytest

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
def clean_env(monkeypatch):
  """Strip every provider key, then set only the two fast mocks.

  mock_slow_provider sleeps 120s — never set its key.
  PROXDASH_API_KEY is stripped so proxdash stays disconnected.
  """
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  for k, v in _MOCK_KEYS.items():
    monkeypatch.setenv(k, v)
  yield


def _proxdash_off() -> types.ProxDashOptions:
  """Proxdash options that short-circuit the /models/configs network fetch."""
  return types.ProxDashOptions(disable_proxdash=True)


def _build_bare_client(**kwargs) -> client.ProxAIClient:
  """Build a ProxAIClient with proxdash disabled (no network).

  Does NOT set run_type=TEST or register a default model — use this for
  construction/property tests that don't call generate().
  """
  kwargs.setdefault('proxdash_options', _proxdash_off())
  return client.ProxAIClient(**kwargs)


def _register_mock_providers(mc_instance: model_configs.ModelConfigs) -> None:
  """Register mock_provider + mock_failing_provider into the given instance.

  The client constructs its own ModelConfigs (bundled v1.3.x registry) that
  doesn't know about our mock providers. Registering them here extends the
  client's live registry — AvailableModels references the SAME object, so
  the mocks become visible to both halves of the client without a swap.
  """
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
                input_token_cost=0.0, output_token_cost=0.0),
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
                    video=NS, multi_modal=NS),
            ),
        )
    )


def _build_client(
    *,
    with_mock_model: bool = True,
    **kwargs,
) -> client.ProxAIClient:
  """Build a ProxAIClient wired for generate() against mock providers.

  Forces run_type=TEST on the underlying AvailableModels (post-init),
  registers mock_provider/mock_failing_provider into the client's existing
  model_configs, and optionally sets the default TEXT model to mock_provider.
  """
  px_client = _build_bare_client(**kwargs)
  px_client.available_models_instance.run_type = types.RunType.TEST
  _register_mock_providers(px_client.model_configs_instance)
  if with_mock_model:
    px_client.set_model(('mock_provider', 'mock_model'))
  return px_client


# =============================================================================
# ProxAIClientParams — dataclass defaults + custom values
# =============================================================================


class TestProxAIClientParams:

  def test_default_values(self):
    params = client.ProxAIClientParams()
    assert params.experiment_path is None
    assert params.cache_options is None
    assert params.logging_options is None
    assert params.proxdash_options is None
    assert params.provider_call_options is None
    assert params.model_probe_options is None
    assert params.debug_options is None

  def test_custom_values_via_nested_options(self):
    cache_opts = types.CacheOptions(cache_path='/tmp/c')
    provider_opts = types.ProviderCallOptions(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        suppress_provider_errors=True,
    )
    probe_opts = types.ModelProbeOptions(
        allow_multiprocessing=False, timeout=30,
    )
    debug_opts = types.DebugOptions(keep_raw_provider_response=True)
    params = client.ProxAIClientParams(
        experiment_path='test/exp',
        cache_options=cache_opts,
        provider_call_options=provider_opts,
        model_probe_options=probe_opts,
        debug_options=debug_opts,
    )
    assert params.experiment_path == 'test/exp'
    assert params.cache_options is cache_opts
    assert params.provider_call_options is provider_opts
    assert params.model_probe_options is probe_opts
    assert params.debug_options is debug_opts


# =============================================================================
# Init — happy paths + state round-trip + conflict errors
# =============================================================================


class TestInit:

  def test_init_from_direct_kwargs_populates_attributes(self):
    px_client = _build_bare_client(experiment_path='exp/a')
    assert px_client.experiment_path == 'exp/a'
    assert px_client.run_type == types.RunType.PRODUCTION
    assert px_client.proxdash_options.disable_proxdash is True

  def test_init_from_params_populates_attributes(self):
    params = client.ProxAIClientParams(
        experiment_path='exp/b',
        proxdash_options=_proxdash_off(),
    )
    px_client = client.ProxAIClient(init_from_params=params)
    assert px_client.experiment_path == 'exp/b'

  def test_init_from_state_round_trip(self):
    original = _build_bare_client(experiment_path='round/trip')
    state = original.get_state()
    restored = client.ProxAIClient(init_from_state=state)
    assert restored.experiment_path == 'round/trip'
    assert restored.proxdash_options.disable_proxdash is True

  def test_init_from_params_with_direct_kwargs_raises(self):
    """Passing init_from_params together with a direct kwarg must raise."""
    with pytest.raises(
        ValueError,
        match='init_from_params or init_from_state cannot be set',
    ):
      client.ProxAIClient(
          experiment_path='x',
          init_from_params=client.ProxAIClientParams(),
      )


# =============================================================================
# Properties — non-trivial setters only
# =============================================================================


class TestProperties:

  def test_experiment_path_validation_rejects_bad_value(self):
    """experiment_path goes through validate_experiment_path()."""
    px_client = _build_bare_client()
    with pytest.raises(ValueError):
      # Leading slash is rejected by validate_experiment_path.
      px_client.experiment_path = '/leading/slash'

  def test_root_logging_path_must_exist(self):
    px_client = _build_bare_client()
    with pytest.raises(ValueError, match='Root logging path does not exist'):
      px_client.root_logging_path = '/this/does/not/exist/anywhere'

  def test_logging_options_joins_experiment_path(self, tmp_path):
    """logging_options setter appends experiment_path to logging_path."""
    px_client = _build_bare_client(experiment_path='exp/deep')
    px_client.logging_options = types.LoggingOptions(logging_path=str(tmp_path))
    assert px_client.logging_options.logging_path == str(
        tmp_path / 'exp' / 'deep'
    )

  def test_cache_options_requires_path_or_disable(self):
    px_client = _build_bare_client()
    with pytest.raises(ValueError, match='cache_path is required'):
      # Neither cache_path nor disable_model_cache → setter must raise.
      px_client.cache_options = types.CacheOptions()

  def test_cache_options_roundtrip(self, tmp_path):
    px_client = _build_bare_client()
    opts = types.CacheOptions(
        cache_path=str(tmp_path), unique_response_limit=7,
    )
    px_client.cache_options = opts
    assert px_client.cache_options.cache_path == str(tmp_path)
    assert px_client.cache_options.unique_response_limit == 7

  def test_proxdash_options_roundtrip(self):
    px_client = _build_bare_client()
    px_client.proxdash_options = types.ProxDashOptions(
        disable_proxdash=True, stdout=True,
    )
    assert px_client.proxdash_options.disable_proxdash is True
    assert px_client.proxdash_options.stdout is True

  def test_provider_call_options_roundtrip(self):
    px_client = _build_bare_client()
    px_client.provider_call_options = types.ProviderCallOptions(
        feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
        suppress_provider_errors=True,
    )
    assert (px_client.provider_call_options.feature_mapping_strategy
            == types.FeatureMappingStrategy.STRICT)
    assert px_client.provider_call_options.suppress_provider_errors is True

  def test_model_probe_options_timeout_must_be_positive(self):
    px_client = _build_bare_client()
    with pytest.raises(ValueError, match='timeout must be >= 1'):
      px_client.model_probe_options = types.ModelProbeOptions(timeout=0)

  def test_debug_options_roundtrip(self):
    px_client = _build_bare_client()
    px_client.debug_options = types.DebugOptions(keep_raw_provider_response=True)
    assert px_client.debug_options.keep_raw_provider_response is True


# =============================================================================
# Cache managers — lazy init, None fallback
# =============================================================================


class TestCacheManagers:

  def test_default_model_cache_manager_always_initialized(self):
    """Without cache_options, model_cache_manager falls back to default."""
    px_client = _build_bare_client()
    assert px_client.default_model_cache_manager is not None
    # Absent cache_options → model_cache_manager == default_model_cache_manager.
    assert (px_client.model_cache_manager
            is px_client.default_model_cache_manager)

  def test_query_cache_manager_is_none_without_cache_options(self):
    px_client = _build_bare_client()
    assert px_client.query_cache_manager is None

  def test_query_cache_manager_lazy_initialized_with_cache_options(
      self, tmp_path,
  ):
    px_client = _build_bare_client(
        cache_options=types.CacheOptions(cache_path=str(tmp_path)),
    )
    assert px_client.query_cache_manager is not None
    assert (px_client.query_cache_manager.status
            == types.QueryCacheManagerStatus.WORKING)


# =============================================================================
# get_default_provider_model — fallback ladder PYDANTIC → JSON → TEXT
# =============================================================================


class TestGetDefaultProviderModel:

  def test_returns_registered_text_model(self):
    px_client = _build_client()  # registers mock_provider as TEXT
    pm = px_client.get_default_provider_model()
    assert pm.provider == 'mock_provider'
    assert pm.model == 'mock_model'

  def test_falls_back_json_to_text(self):
    """No JSON model registered → PYDANTIC request falls back to JSON → TEXT."""
    px_client = _build_client()  # only TEXT registered
    pm = px_client.get_default_provider_model(
        output_format_type=types.OutputFormatType.PYDANTIC,
    )
    # Fallback ladder PYDANTIC → JSON → TEXT lands on mock_provider.
    assert pm.provider == 'mock_provider'

  def test_unsupported_output_format_raises(self):
    px_client = _build_client(with_mock_model=False)
    with pytest.raises(ValueError, match='Output format type not supported'):
      px_client.get_default_provider_model(
          output_format_type=types.OutputFormatType.IMAGE,
      )


# =============================================================================
# set_model — variants + error paths
# =============================================================================


class TestSetModel:

  def test_positional_tuple_registers_text(self):
    px_client = _build_client(with_mock_model=False)
    px_client.set_model(('mock_provider', 'mock_model'))
    pm = px_client.registered_models[types.OutputFormatType.TEXT]
    assert pm.provider == 'mock_provider'

  def test_generate_text_kwarg_registers_text(self):
    px_client = _build_client(with_mock_model=False)
    px_client.set_model(generate_text=('mock_provider', 'mock_model'))
    assert (types.OutputFormatType.TEXT in px_client.registered_models)

  def test_per_format_kwargs_register_each_format(self):
    px_client = _build_client(with_mock_model=False)
    # mock_provider supports JSON/PYDANTIC output formats too (see conftest).
    px_client.set_model(
        generate_text=('mock_provider', 'mock_model'),
        generate_json=('mock_provider', 'mock_model'),
    )
    assert types.OutputFormatType.TEXT in px_client.registered_models
    assert types.OutputFormatType.JSON in px_client.registered_models

  def test_no_args_raises(self):
    px_client = _build_client(with_mock_model=False)
    with pytest.raises(ValueError, match='At least one model must be'):
      px_client.set_model()

  def test_provider_model_plus_generate_text_raises(self):
    px_client = _build_client(with_mock_model=False)
    with pytest.raises(
        ValueError, match='provider_model and generate_text cannot',
    ):
      px_client.set_model(
          provider_model=('mock_provider', 'mock_model'),
          generate_text=('mock_provider', 'mock_model'),
      )


# =============================================================================
# generate() validation — the six error branches
# =============================================================================


class TestGenerateValidation:

  def test_prompt_and_messages_together_raises(self):
    px_client = _build_client()
    with pytest.raises(
        ValueError, match='prompt and messages cannot be used together',
    ):
      px_client.generate(prompt='hi', messages=[{'role': 'user', 'content': 'x'}])

  def test_system_prompt_and_messages_together_raises(self):
    px_client = _build_client()
    with pytest.raises(
        ValueError, match='system_prompt and messages cannot be used',
    ):
      px_client.generate(
          system_prompt='you are helpful',
          messages=[{'role': 'user', 'content': 'x'}],
      )

  def test_suppress_and_fallback_models_together_raises(self):
    px_client = _build_client()
    with pytest.raises(
        ValueError, match='suppress_provider_errors and fallback_models',
    ):
      px_client.generate(
          prompt='hi',
          connection_options=types.ConnectionOptions(
              suppress_provider_errors=True,
              fallback_models=[('mock_provider', 'mock_model')],
          ),
      )

  def test_endpoint_and_fallback_models_together_raises(self):
    px_client = _build_client()
    with pytest.raises(
        ValueError, match='endpoint and fallback_models cannot be used',
    ):
      px_client.generate(
          prompt='hi',
          connection_options=types.ConnectionOptions(
              endpoint='generate.text',
              fallback_models=[('mock_provider', 'mock_model')],
          ),
      )

  def test_override_cache_value_without_cache_raises(self):
    """override_cache_value=True without a working query cache must raise."""
    px_client = _build_client()  # no cache_options
    with pytest.raises(
        ValueError, match='override_cache_value is True but query cache',
    ):
      px_client.generate(
          prompt='hi',
          connection_options=types.ConnectionOptions(override_cache_value=True),
      )

  def test_no_resolvable_default_model_raises(self, monkeypatch):
    """No registered model + no working model in env → helpful error.

    Stripping the mock keys means the api-key filter drops every mock model
    before _test_models can classify it as working, so get_default_provider_model
    exhausts the priority list + any-working-model fallback → raises.
    """
    monkeypatch.delenv('MOCK_PROVIDER_API_KEY', raising=False)
    monkeypatch.delenv('MOCK_FAILING_PROVIDER', raising=False)
    px_client = _build_client(with_mock_model=False)
    with pytest.raises(
        ValueError, match='no default model could be resolved',
    ):
      px_client.generate(prompt='hi')


# =============================================================================
# Alias smoke — generate_text / generate_json / generate_pydantic
# =============================================================================


class _PydanticCity:
  """Dummy pydantic-ish class for generate_pydantic smoke."""
  # (Using a dataclass-like stub would need BaseModel — keep it minimal;
  # mock_provider returns a fixed JSON-compatible dict that we don't validate.)


class TestGenerateAliasSmoke:

  def test_generate_text_returns_string(self):
    px_client = _build_client()
    result = px_client.generate_text(prompt='hello')
    assert isinstance(result, str)
    assert result == 'mock response'

  def test_generate_json_returns_dict(self):
    px_client = _build_client()
    result = px_client.generate_json(prompt='hello')
    # mock_provider returns {"name": "John Doe", "age": 30} for JSON format.
    assert isinstance(result, dict)
    assert result == {'name': 'John Doe', 'age': 30}

  def test_generate_pydantic_returns_instance(self):
    """Smoke: call succeeds and returns a parsed result (not an error string).

    The mock provider returns a JSON dict; full pydantic validation is the
    responsibility of ResultAdapter and is tested there. We only verify the
    alias wiring runs end-to-end.
    """
    import pydantic

    class _City(pydantic.BaseModel):
      name: str
      age: int

    px_client = _build_client()
    result = px_client.generate_pydantic(
        prompt='hello',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC, pydantic_class=_City,
        ),
    )
    assert isinstance(result, _City)
    assert result.name == 'John Doe'


# =============================================================================
# fallback_models — happy path
# =============================================================================


class TestGenerateFallback:

  def test_fallback_models_uses_next_after_failure(self):
    """First provider fails → second (working) succeeds, returns its result."""
    px_client = _build_client(with_mock_model=False)
    px_client.set_model(('mock_failing_provider', 'mock_failing_model'))
    result = px_client.generate(
        prompt='hi',
        connection_options=types.ConnectionOptions(
            fallback_models=[('mock_provider', 'mock_model')],
        ),
    )
    # The fallback model is mock_provider, which succeeds.
    assert result.result.status == types.ResultStatusType.SUCCESS
    assert result.query.provider_model.provider == 'mock_provider'


# =============================================================================
# get_current_options / get_state
# =============================================================================


class TestGetCurrentOptions:

  def test_returns_run_options_with_nested_shape(self):
    px_client = _build_bare_client(experiment_path='exp/snapshot')
    options = px_client.get_current_options()
    assert isinstance(options, types.RunOptions)
    assert options.experiment_path == 'exp/snapshot'
    # Nested options present (not as flat fields — those were removed).
    assert isinstance(options.provider_call_options, types.ProviderCallOptions)
    assert isinstance(options.model_probe_options, types.ModelProbeOptions)
    assert isinstance(options.debug_options, types.DebugOptions)

  def test_returns_dict_when_json_true(self):
    px_client = _build_bare_client()
    options = px_client.get_current_options(json=True)
    assert isinstance(options, dict)
    assert 'run_type' in options
    # Verify nested shape is serialized too.
    assert 'provider_call_options' in options


class TestGetState:

  def test_get_state_snapshots_nested_options(self):
    px_client = _build_bare_client(
        provider_call_options=types.ProviderCallOptions(
            suppress_provider_errors=True,
        ),
    )
    state = px_client.get_state()
    assert isinstance(state, types.ProxAIClientState)
    # Restore and verify the nested option survived the round trip.
    restored = client.ProxAIClient(init_from_state=state)
    assert restored.provider_call_options.suppress_provider_errors is True


# =============================================================================
# ModelConnector + FileConnector — delegation smoke
# =============================================================================


class TestModelConnector:

  def test_models_is_bound_to_this_client(self):
    px_client = _build_client()
    assert px_client.models._client_getter() is px_client

  def test_models_list_models_delegates(self):
    """Smoke that px_client.models.list_models() reaches AvailableModels."""
    px_client = _build_client()
    models = px_client.models.list_models(recommended_only=False)
    providers = {m.provider for m in models}
    assert 'mock_provider' in providers

  def test_distinct_clients_have_distinct_models_connectors(self):
    a = _build_bare_client()
    b = _build_bare_client()
    assert a.models is not b.models
    assert a.models._client_getter() is a
    assert b.models._client_getter() is b


class TestFileConnector:

  def test_files_is_bound_to_this_client(self):
    px_client = _build_bare_client()
    assert px_client.files._client_getter() is px_client

  def test_files_is_download_supported_delegates(self):
    """Smoke that px_client.files.is_download_supported() reaches files_manager."""
    px_client = _build_bare_client()
    # gemini does not support download — boolean False expected.
    assert px_client.files.is_download_supported('gemini') is False


# =============================================================================
# keep_raw_provider_response — debug escape hatch
# =============================================================================


class TestKeepRawProviderResponse:

  def test_keep_raw_with_cache_options_raises(self, tmp_path):
    """The two settings are mutually exclusive per _validate_raw_...()."""
    with pytest.raises(
        ValueError, match='keep_raw_provider_response=True is incompatible',
    ):
      _build_bare_client(
          cache_options=types.CacheOptions(cache_path=str(tmp_path)),
          debug_options=types.DebugOptions(keep_raw_provider_response=True),
      )

  def test_keep_raw_without_cache_is_ok(self):
    px_client = _build_bare_client(
        debug_options=types.DebugOptions(keep_raw_provider_response=True),
    )
    assert px_client.debug_options.keep_raw_provider_response is True

  def test_keep_raw_attaches_raw_to_call_record(self):
    """Success path: raw provider response lands on call_record.debug."""
    px_client = _build_client(
        debug_options=types.DebugOptions(keep_raw_provider_response=True),
    )
    result = px_client.generate(prompt='hi')
    assert result.debug is not None
    assert result.debug.raw_provider_response == 'mock response'

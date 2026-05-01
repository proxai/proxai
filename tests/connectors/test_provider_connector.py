"""Tests for ProviderConnector.

The request-pipeline base class. Every provider subclass inherits this;
generate() is the one public entry point. Tests are organized by
responsibility:

  - Subclass contract validation (__init_subclass__)
  - Token map validation
  - Helpers: JSON extraction, token count, cost estimation
  - Safe execution wrapper
  - ExecutorResult + keep_raw_provider_response contract
  - Endpoint selection
  - Cache integration
  - Auto-upload media
  - generate() orchestration (the main public entry point)
  - Feature-tag rollup introspection
"""

import datetime
import json
from unittest import mock

import pydantic
import pytest

import proxai.caching.query_cache as query_cache
import proxai.chat.chat_session as chat_session
import proxai.chat.message as message_module
import proxai.chat.message_content as message_content
import proxai.connectors.provider_connector as provider_connector
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types


class _SamplePydanticModel(pydantic.BaseModel):
  """Minimal pydantic model for cache reconstruction tests."""

  name: str
  age: int


@pytest.fixture(autouse=True)
def setup_test(monkeypatch, requests_mock):
  """Clean env keys and stub the ProxDash key-verification endpoint."""
  import proxai.connectors.model_configs as model_configs
  monkeypatch.delenv('PROXDASH_API_KEY', raising=False)
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.setenv(api_key, 'test_api_key')
  requests_mock.get(
      'https://proxainest-production.up.railway.app/ingestion/verify-key',
      text='{"success": true, "data": {"permission": "ALL"}}',
      status_code=200,
  )
  yield


# =============================================================================
# Shared test helpers
# =============================================================================


def _build_connector(
    connector_cls,
    *,
    feature_mapping_strategy=None,
    suppress_provider_errors=False,
    keep_raw_provider_response=False,
    query_cache_manager=None,
    files_manager=None,
    proxdash_connection=None,
    provider_token_value_map=None,
):
  """Construct a ProviderConnector with test-friendly defaults.

  Derives the token map from connector_cls.PROVIDER_API_KEYS when omitted.
  Wraps feature_mapping_strategy / suppress_provider_errors in
  ProviderCallOptions and keep_raw_provider_response in DebugOptions to
  match the current ProviderConnectorParams shape.
  """
  if provider_token_value_map is None:
    provider_token_value_map = dict.fromkeys(connector_cls.PROVIDER_API_KEYS, 'unused')
  pco_kwargs = {'suppress_provider_errors': suppress_provider_errors}
  if feature_mapping_strategy is not None:
    pco_kwargs['feature_mapping_strategy'] = feature_mapping_strategy
  params = provider_connector.ProviderConnectorParams(
      run_type=types.RunType.TEST,
      provider_call_options=types.ProviderCallOptions(**pco_kwargs),
      logging_options=types.LoggingOptions(),
      provider_token_value_map=provider_token_value_map,
      debug_options=types.DebugOptions(
          keep_raw_provider_response=keep_raw_provider_response,
      ),
      query_cache_manager=query_cache_manager,
      files_manager=files_manager,
      proxdash_connection=proxdash_connection,
  )
  return connector_cls(init_from_params=params)


def _make_query_record(
    prompt='hi',
    output_format=None,
    connection_options=None,
    **kwargs,
):
  if output_format is None:
    output_format = types.OutputFormat(type=types.OutputFormatType.TEXT)
  return types.QueryRecord(
      prompt=prompt,
      output_format=output_format,
      connection_options=connection_options,
      **kwargs,
  )


def _make_provider_model_config(
    provider='capture_test_provider',
    model='capture_test_model',
    features=None,
    pricing=None,
    metadata=None,
):
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider=provider,
          model=model,
          provider_model_identifier=model,
      ),
      pricing=pricing if pricing is not None else types.ProviderModelPricingType(),
      features=features if features is not None else types.FeatureConfigType(),
      metadata=metadata if metadata is not None else types.ProviderModelMetadataType(),
  )


def _make_query_cache_manager(tmp_path):
  """Build a QueryCacheManager rooted at tmp_path with minimal config."""
  return query_cache.QueryCacheManager(
      init_from_params=query_cache.QueryCacheManagerParams(
          cache_options=types.CacheOptions(
              cache_path=str(tmp_path),
              unique_response_limit=1,
          ),
          shard_count=3,
          response_per_file=4,
          cache_response_size=10,
      )
  )


# =============================================================================
# ExecutorResult / keep_raw_provider_response contract
# =============================================================================


_RAW_SUCCESS_SENTINEL = {'sentinel': 'raw_success_response'}


class _CaptureTestConnector(provider_connector.ProviderConnector):
  """Minimal ProviderConnector with a single executor we control."""

  PROVIDER_NAME = 'capture_test_provider'
  PROVIDER_API_KEYS = ['CAPTURE_TEST_KEY']
  ENDPOINT_PRIORITY = ['generate.text']
  ENDPOINT_CONFIG = {
      'generate.text':
          types.FeatureConfigType(prompt=types.FeatureSupportType.SUPPORTED,),
  }
  ENDPOINT_EXECUTORS = {
      'generate.text': '_generate_text_executor',
  }

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _generate_text_executor(self, query_record):

    def _runner():
      return _RAW_SUCCESS_SENTINEL

    response, result_record = self._safe_provider_query(_runner)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response
    )


class _FailingCaptureTestConnector(provider_connector.ProviderConnector):
  """Minimal connector whose executor raises inside _safe_provider_query."""

  PROVIDER_NAME = 'capture_failing_test_provider'
  PROVIDER_API_KEYS = ['CAPTURE_TEST_KEY']
  ENDPOINT_PRIORITY = ['generate.text']
  ENDPOINT_CONFIG = {
      'generate.text':
          types.FeatureConfigType(prompt=types.FeatureSupportType.SUPPORTED,),
  }
  ENDPOINT_EXECUTORS = {
      'generate.text': '_generate_text_executor',
  }

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _generate_text_executor(self, query_record):

    def _runner():
      raise RuntimeError('forced failure for capture test')

    response, result_record = self._safe_provider_query(_runner)
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response
    )


class TestExecutorResultContract:

  def test_success_carries_raw_response(self):
    connector = _build_connector(
        _CaptureTestConnector, keep_raw_provider_response=True
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=_make_query_record(),
        provider_model_config=_make_provider_model_config(),
    )
    assert isinstance(executor_result, types.ExecutorResult)
    assert executor_result.raw_provider_response is _RAW_SUCCESS_SENTINEL
    assert executor_result.result_record.status == types.ResultStatusType.SUCCESS

  def test_success_without_flag_still_returns_raw(self):
    connector = _build_connector(
        _CaptureTestConnector, keep_raw_provider_response=False
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=_make_query_record(),
        provider_model_config=_make_provider_model_config(),
    )
    assert executor_result.raw_provider_response is _RAW_SUCCESS_SENTINEL
    assert executor_result.result_record.status == types.ResultStatusType.SUCCESS

  def test_failure_returns_none_raw_response(self):
    connector = _build_connector(
        _FailingCaptureTestConnector, keep_raw_provider_response=True
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=_make_query_record(),
        provider_model_config=_make_provider_model_config(),
    )
    assert executor_result.raw_provider_response is None
    assert executor_result.result_record.status == types.ResultStatusType.FAILED


# =============================================================================
# __init_subclass__ contract
# =============================================================================


class TestInitSubclassContract:
  """Every ProviderConnector subclass must declare the 5 required attrs
  and ENDPOINT_PRIORITY must key-match both ENDPOINT_CONFIG and
  ENDPOINT_EXECUTORS. These checks fire at class-definition time.
  """

  def test_missing_provider_name(self):
    with pytest.raises(TypeError, match='must define PROVIDER_NAME'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_API_KEYS = []
        ENDPOINT_PRIORITY = ['e']
        ENDPOINT_CONFIG = {'e': types.FeatureConfigType()}
        ENDPOINT_EXECUTORS = {'e': 'fn'}

  def test_missing_provider_api_keys(self):
    with pytest.raises(TypeError, match='must define PROVIDER_API_KEYS'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        ENDPOINT_PRIORITY = ['e']
        ENDPOINT_CONFIG = {'e': types.FeatureConfigType()}
        ENDPOINT_EXECUTORS = {'e': 'fn'}

  def test_missing_endpoint_priority(self):
    with pytest.raises(TypeError, match='must define ENDPOINT_PRIORITY'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        PROVIDER_API_KEYS = []
        ENDPOINT_CONFIG = {'e': types.FeatureConfigType()}
        ENDPOINT_EXECUTORS = {'e': 'fn'}

  def test_missing_endpoint_config(self):
    with pytest.raises(TypeError, match='must define ENDPOINT_CONFIG'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        PROVIDER_API_KEYS = []
        ENDPOINT_PRIORITY = ['e']
        ENDPOINT_EXECUTORS = {'e': 'fn'}

  def test_missing_endpoint_executors(self):
    with pytest.raises(TypeError, match='must define ENDPOINT_EXECUTORS'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        PROVIDER_API_KEYS = []
        ENDPOINT_PRIORITY = ['e']
        ENDPOINT_CONFIG = {'e': types.FeatureConfigType()}

  def test_priority_vs_config_key_mismatch(self):
    with pytest.raises(ValueError, match='ENDPOINT_PRIORITY and ENDPOINT_CONFIG'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        PROVIDER_API_KEYS = []
        ENDPOINT_PRIORITY = ['a']
        ENDPOINT_CONFIG = {'b': types.FeatureConfigType()}
        ENDPOINT_EXECUTORS = {'a': 'fn'}

  def test_priority_vs_executors_key_mismatch(self):
    with pytest.raises(
        ValueError, match='ENDPOINT_PRIORITY and ENDPOINT_EXECUTORS'):
      class _(provider_connector.ProviderConnector):
        PROVIDER_NAME = 'x'
        PROVIDER_API_KEYS = []
        ENDPOINT_PRIORITY = ['a']
        ENDPOINT_CONFIG = {'a': types.FeatureConfigType()}
        ENDPOINT_EXECUTORS = {'b': 'fn'}


# =============================================================================
# _validate_provider_token_value_map
# =============================================================================


class TestValidateProviderTokenValueMap:

  def test_none_token_map_raises(self):
    params = provider_connector.ProviderConnectorParams(
        run_type=types.RunType.TEST,
        provider_call_options=types.ProviderCallOptions(),
        logging_options=types.LoggingOptions(),
        provider_token_value_map=None,
        debug_options=types.DebugOptions(),
    )
    with pytest.raises(ValueError, match='needs to be set'):
      _CaptureTestConnector(init_from_params=params)

  def test_missing_required_key_raises(self):
    with pytest.raises(
        ValueError, match='needs to contain CAPTURE_TEST_KEY'):
      _build_connector(
          _CaptureTestConnector,
          provider_token_value_map={'SOME_OTHER_KEY': 'x'},
      )


# =============================================================================
# _extract_json_from_text — four-strategy JSON recovery
# =============================================================================


class TestExtractJsonFromText:

  def test_direct_json_parse(self):
    c = _build_connector(_CaptureTestConnector)
    assert c._extract_json_from_text('{"a": 1, "b": [2, 3]}') == {
        'a': 1, 'b': [2, 3]
    }

  def test_markdown_json_fence(self):
    c = _build_connector(_CaptureTestConnector)
    assert c._extract_json_from_text('```json\n{"a": 1}\n```') == {'a': 1}
    # Also works without the `json` hint:
    assert c._extract_json_from_text('```\n{"a": 1}\n```') == {'a': 1}

  def test_find_braces_in_prose(self):
    c = _build_connector(_CaptureTestConnector)
    assert c._extract_json_from_text(
        'Here is the output: {"a": 1} thanks.'
    ) == {'a': 1}

  def test_single_quoted_python_dict(self):
    c = _build_connector(_CaptureTestConnector)
    # Python repr-style single quotes; stripped if no double-quotes exist.
    assert c._extract_json_from_text("{'a': 1, 'b': 2}") == {'a': 1, 'b': 2}

  def test_unparseable_raises(self):
    c = _build_connector(_CaptureTestConnector)
    with pytest.raises(json.JSONDecodeError):
      c._extract_json_from_text('not json at all')


# =============================================================================
# get_token_count_estimate
# =============================================================================


class TestGetTokenCountEstimate:

  def test_empty_returns_zero(self):
    c = _build_connector(_CaptureTestConnector)
    assert c.get_token_count_estimate() == 0
    assert c.get_token_count_estimate(None) == 0
    assert c.get_token_count_estimate([]) == 0

  def test_dict_via_json_dumps(self):
    c = _build_connector(_CaptureTestConnector)
    # dict goes through json.dumps → non-zero token count.
    assert c.get_token_count_estimate({'key': 'value'}) > 0

  def test_text_and_json_blocks_sum_image_contributes_zero(self):
    c = _build_connector(_CaptureTestConnector)
    text_block = message_content.MessageContent(
        type=message_content.ContentType.TEXT, text='hello world')
    json_block = message_content.MessageContent(
        type=message_content.ContentType.JSON, json={'k': 'v'})
    image_block = message_content.MessageContent(
        type=message_content.ContentType.IMAGE,
        source='https://example.com/a.png')
    text_only = c.get_token_count_estimate([text_block])
    json_only = c.get_token_count_estimate([json_block])
    combined = c.get_token_count_estimate([text_block, json_block, image_block])
    assert text_only > 0 and json_only > 0
    assert combined == text_only + json_only  # image contributes 0

  def test_unknown_content_type_raises(self):
    c = _build_connector(_CaptureTestConnector)

    class _FakeType:
      value = 'unknown'

    class _FakeMessage:
      type = _FakeType()

    with pytest.raises(ValueError, match='Invalid message type'):
      c.get_token_count_estimate([_FakeMessage()])


# =============================================================================
# get_estimated_cost
# =============================================================================


class TestGetEstimatedCost:

  def test_computes_sum_of_products(self):
    c = _build_connector(_CaptureTestConnector)
    call_record = types.CallRecord(
        result=types.ResultRecord(
            usage=types.UsageType(input_tokens=100, output_tokens=200),
        ),
    )
    pmc = _make_provider_model_config(
        pricing=types.ProviderModelPricingType(
            input_token_cost=10,
            output_token_cost=20,
        )
    )
    # 100*10 + 200*20 = 1000 + 4000 = 5000 nano-USD.
    assert c.get_estimated_cost(call_record, pmc) == 5000

  def test_none_values_treated_as_zero(self):
    c = _build_connector(_CaptureTestConnector)
    call_record = types.CallRecord(
        result=types.ResultRecord(
            usage=types.UsageType(input_tokens=None, output_tokens=100),
        ),
    )
    pmc = _make_provider_model_config(
        pricing=types.ProviderModelPricingType(
            input_token_cost=None,
            output_token_cost=20,
        )
    )
    # input 0 * anything = 0; output 100 * 20 = 2000. Sum=2000.
    assert c.get_estimated_cost(call_record, pmc) == 2000

  def test_nano_usd_unit_contract_realistic_price(self):
    """Lock the nano-USD unit: Claude Haiku @ $0.80 / 1M tokens input.

    Per ProviderModelPricingType, $0.80 / 1M tokens = 800 nano-USD per
    token. One million input tokens should therefore cost exactly
    800_000_000 nano-USD = $0.80.
    """
    c = _build_connector(_CaptureTestConnector)
    call_record = types.CallRecord(
        result=types.ResultRecord(
            usage=types.UsageType(
                input_tokens=1_000_000, output_tokens=0,
            ),
        ),
    )
    pmc = _make_provider_model_config(
        pricing=types.ProviderModelPricingType(
            input_token_cost=800,
            output_token_cost=4000,
        )
    )
    assert c.get_estimated_cost(call_record, pmc) == 800_000_000


# =============================================================================
# _safe_provider_query
# =============================================================================


class TestSafeProviderQuery:

  def test_success_returns_value_and_success_record(self):
    c = _build_connector(_CaptureTestConnector)
    response, rr = c._safe_provider_query(lambda: 'ok')
    assert response == 'ok'
    assert rr.status == types.ResultStatusType.SUCCESS
    assert rr.role == types.MessageRoleType.ASSISTANT
    assert rr.error is None

  def test_exception_returns_none_and_failure_record_with_traceback(self):
    c = _build_connector(_CaptureTestConnector)

    def _boom():
      raise RuntimeError('boom')

    response, rr = c._safe_provider_query(_boom)
    assert response is None
    assert rr.status == types.ResultStatusType.FAILED
    assert rr.role == types.MessageRoleType.ASSISTANT
    assert isinstance(rr.error, RuntimeError)
    assert 'RuntimeError' in rr.error_traceback


# =============================================================================
# _execute_call runs ResultAdapter on success
# =============================================================================


class _TextContentConnector(provider_connector.ProviderConnector):
  """Connector whose executor returns a TEXT content block — used to
  verify that _execute_call runs ResultAdapter when the query asks for
  JSON output.
  """

  PROVIDER_NAME = 'text_content_provider'
  PROVIDER_API_KEYS = []
  ENDPOINT_PRIORITY = ['chat']
  ENDPOINT_CONFIG = {
      'chat': types.FeatureConfigType(
          prompt=types.FeatureSupportType.SUPPORTED,
      ),
  }
  ENDPOINT_EXECUTORS = {'chat': '_chat_executor'}

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _chat_executor(self, query_record):
    rr = types.ResultRecord(
        status=types.ResultStatusType.SUCCESS,
        role=types.MessageRoleType.ASSISTANT,
        content=[
            message_content.MessageContent(
                type=message_content.ContentType.TEXT,
                text='{"result": "ok"}',
            ),
        ],
    )
    return types.ExecutorResult(result_record=rr)


class TestExecuteCallAdaptsResult:

  def test_execute_call_runs_result_adapter_on_success(self):
    c = _build_connector(_TextContentConnector)
    query_record = _make_query_record(
        output_format=types.OutputFormat(type=types.OutputFormatType.JSON),
    )
    pmc = _make_provider_model_config(
        provider='text_content_provider', model='m',
    )
    result = c._execute_call(
        chosen_executor=c._chat_executor,
        chosen_endpoint='chat',
        query_record=query_record,
        provider_model_config=pmc,
    )
    # ResultAdapter transformed TEXT→JSON because query asked for JSON.
    assert result.result_record.content[0].type == (
        message_content.ContentType.JSON
    )
    assert result.result_record.content[0].json == {'result': 'ok'}


# =============================================================================
# Endpoint selection — _find_compatible_endpoint, _prepare_execution
# =============================================================================


_S = types.FeatureSupportType.SUPPORTED
_BE = types.FeatureSupportType.BEST_EFFORT
_NS = types.FeatureSupportType.NOT_SUPPORTED


def _endpoint_config(prompt_support):
  """A FeatureConfigType with the given prompt support and fully-enabled
  output format (so the only support variable is prompt).
  """
  return types.FeatureConfigType(
      prompt=prompt_support,
      output_format=types.OutputFormatConfigType(text=_S),
  )


def _all_supported_features():
  return types.FeatureConfigType(
      prompt=_S, output_format=types.OutputFormatConfigType(text=_S),
  )


def _make_connector_with_endpoints(endpoint_support_map):
  """Build a ProviderConnector subclass where each endpoint's prompt
  support is as specified. Insertion order = ENDPOINT_PRIORITY order.
  """
  priority = list(endpoint_support_map.keys())
  endpoint_config = {
      ep: _endpoint_config(support)
      for ep, support in endpoint_support_map.items()
  }

  class _DynConnector(provider_connector.ProviderConnector):
    PROVIDER_NAME = 'dyn_ep_provider'
    PROVIDER_API_KEYS = []
    ENDPOINT_PRIORITY = priority
    ENDPOINT_CONFIG = endpoint_config
    ENDPOINT_EXECUTORS = dict.fromkeys(priority, '_noop_exec')

    def init_model(self):
      return None

    def init_mock_model(self):
      return None

    def _noop_exec(self, query_record):
      return types.ExecutorResult(result_record=types.ResultRecord())

  return _DynConnector


class TestFindCompatibleEndpoint:

  def test_supported_endpoint_picked_first(self):
    cls = _make_connector_with_endpoints({'first': _S, 'second': _S})
    c = _build_connector(cls)
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    assert c._find_compatible_endpoint(_make_query_record(), pmc) == 'first'

  def test_best_effort_picked_when_mode_is_best_effort(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _BE})
    c = _build_connector(
        cls,
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
    )
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    assert c._find_compatible_endpoint(_make_query_record(), pmc) == 'second'

  def test_best_effort_raises_in_strict_mode(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _BE})
    c = _build_connector(
        cls, feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
    )
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    qr = _make_query_record(provider_model=pmc.provider_model)
    with pytest.raises(ValueError, match='No compatible endpoint'):
      c._find_compatible_endpoint(qr, pmc)

  def test_all_not_supported_raises(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _NS})
    c = _build_connector(
        cls,
        feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
    )
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    qr = _make_query_record(provider_model=pmc.provider_model)
    with pytest.raises(ValueError, match='No compatible endpoint'):
      c._find_compatible_endpoint(qr, pmc)


class TestPrepareExecution:

  def test_explicit_endpoint_supported_used(self):
    cls = _make_connector_with_endpoints({'first': _S, 'second': _S})
    c = _build_connector(cls)
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    qr = _make_query_record(
        connection_options=types.ConnectionOptions(endpoint='second'),
    )
    executor, endpoint, _ = c._prepare_execution(
        query_record=qr, provider_model_config=pmc,
    )
    assert endpoint == 'second'
    assert executor == c._noop_exec

  def test_explicit_endpoint_not_supported_raises(self):
    cls = _make_connector_with_endpoints({'first': _S, 'second': _NS})
    c = _build_connector(cls)
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    qr = _make_query_record(
        connection_options=types.ConnectionOptions(endpoint='second'),
    )
    with pytest.raises(ValueError, match='not supported'):
      c._prepare_execution(query_record=qr, provider_model_config=pmc)

  def test_explicit_endpoint_best_effort_in_strict_raises(self):
    cls = _make_connector_with_endpoints({'first': _S, 'second': _BE})
    c = _build_connector(
        cls, feature_mapping_strategy=types.FeatureMappingStrategy.STRICT,
    )
    pmc = _make_provider_model_config(
        provider='dyn_ep_provider', features=_all_supported_features(),
    )
    qr = _make_query_record(
        connection_options=types.ConnectionOptions(endpoint='second'),
    )
    with pytest.raises(ValueError, match='STRICT mode'):
      c._prepare_execution(query_record=qr, provider_model_config=pmc)


# =============================================================================
# Cache integration — _get_cached_result, _update_cache,
# _reconstruct_pydantic_from_cache
# =============================================================================


def _make_result_record(output_text='cached', content=None):
  """A ResultRecord with timestamp populated (required by _get_cached_result
  timestamp-refresh logic).
  """
  now = datetime.datetime.now(datetime.timezone.utc)
  return types.ResultRecord(
      status=types.ResultStatusType.SUCCESS,
      role=types.MessageRoleType.ASSISTANT,
      output_text=output_text,
      content=content,
      timestamp=types.TimeStampType(
          start_utc_date=now,
          end_utc_date=now,
          response_time=datetime.timedelta(seconds=1),
          local_time_offset_minute=0,
      ),
  )


class TestGetCachedResult:

  def test_cache_hit_returns_result_with_refreshed_timestamps(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='hi')
    stored_result = _make_result_record(output_text='cached')
    past_time = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    stored_result.timestamp.end_utc_date = past_time
    cache_mgr.cache(query_record=qr, result_record=stored_result)

    result = c._get_cached_result(
        query_record=qr, connection_options=types.ConnectionOptions(),
    )
    assert isinstance(result, types.ResultRecord)
    assert result.output_text == 'cached'
    # Timestamp refreshed to recent.
    assert result.timestamp.end_utc_date > past_time
    # cache_response_time records the cache lookup duration.
    assert result.timestamp.cache_response_time is not None
    assert result.timestamp.cache_response_time >= datetime.timedelta(0)

  def test_cache_miss_returns_fail_reason(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='never-cached')

    result = c._get_cached_result(
        query_record=qr, connection_options=types.ConnectionOptions(),
    )
    assert isinstance(result, types.CacheLookFailReason)

  def test_skip_cache_returns_none_even_if_cached(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='hi')
    cache_mgr.cache(query_record=qr, result_record=_make_result_record())

    assert c._get_cached_result(
        query_record=qr,
        connection_options=types.ConnectionOptions(skip_cache=True),
    ) is None

  def test_override_cache_value_bypasses_read(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='hi')
    cache_mgr.cache(query_record=qr, result_record=_make_result_record())

    assert c._get_cached_result(
        query_record=qr,
        connection_options=types.ConnectionOptions(override_cache_value=True),
    ) is None


class TestUpdateCache:

  def test_update_writes_to_cache(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='hi')
    call_record = types.CallRecord(
        query=qr, result=_make_result_record(output_text='fresh'),
    )
    c._update_cache(
        call_record=call_record, connection_options=types.ConnectionOptions(),
    )
    # Subsequent lookup finds it.
    result = c._get_cached_result(
        query_record=qr, connection_options=types.ConnectionOptions(),
    )
    assert isinstance(result, types.ResultRecord)
    assert result.output_text == 'fresh'

  def test_skip_cache_disables_write(self, tmp_path):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(_CaptureTestConnector, query_cache_manager=cache_mgr)
    qr = _make_query_record(prompt='hi')
    call_record = types.CallRecord(
        query=qr, result=_make_result_record(output_text='fresh'),
    )
    c._update_cache(
        call_record=call_record,
        connection_options=types.ConnectionOptions(skip_cache=True),
    )
    # Lookup misses because nothing was written.
    result = c._get_cached_result(
        query_record=qr, connection_options=types.ConnectionOptions(),
    )
    assert isinstance(result, types.CacheLookFailReason)


class TestReconstructPydanticFromCache:

  def test_reconstructs_instance_value_and_fills_output_pydantic(self):
    c = _build_connector(_CaptureTestConnector)
    qr = _make_query_record(
        output_format=types.OutputFormat(
            type=types.OutputFormatType.PYDANTIC,
            pydantic_class=_SamplePydanticModel,
        ),
    )
    # Cached content block has only instance_json_value — as if round-tripped
    # through JSON storage.
    result_record = types.ResultRecord(
        content=[
            message_content.MessageContent(
                type=message_content.ContentType.PYDANTIC_INSTANCE,
                pydantic_content=message_content.PydanticContent(
                    class_name='_SamplePydanticModel',
                    instance_json_value={'name': 'Alice', 'age': 30},
                ),
            ),
        ],
    )
    c._reconstruct_pydantic_from_cache(qr, result_record)

    pc = result_record.content[0].pydantic_content
    assert isinstance(pc.instance_value, _SamplePydanticModel)
    assert pc.instance_value.name == 'Alice'
    assert pc.instance_value.age == 30
    assert pc.class_value is _SamplePydanticModel
    assert result_record.output_pydantic is pc.instance_value

  def test_noop_when_output_format_not_pydantic(self):
    c = _build_connector(_CaptureTestConnector)
    qr = _make_query_record(
        output_format=types.OutputFormat(type=types.OutputFormatType.TEXT),
    )
    result_record = types.ResultRecord(
        content=[
            message_content.MessageContent(
                type=message_content.ContentType.PYDANTIC_INSTANCE,
                pydantic_content=message_content.PydanticContent(
                    class_name='_SamplePydanticModel',
                    instance_json_value={'name': 'Bob', 'age': 25},
                ),
            ),
        ],
    )
    c._reconstruct_pydantic_from_cache(qr, result_record)
    # No reconstruction happened.
    assert result_record.content[0].pydantic_content.instance_value is None
    assert result_record.output_pydantic is None


# =============================================================================
# _auto_upload_media
# =============================================================================


class _StubFilesManager(state_controller.BaseStateControlled):
  """Minimal FilesManager stub — records upload calls."""

  def __init__(self, is_upload_supported_return=True):
    self.is_upload_supported_return = is_upload_supported_return
    self.upload_calls = []  # list of (media, providers)

  def is_upload_supported(self, media, provider):
    return self.is_upload_supported_return

  def upload(self, media, providers):
    self.upload_calls.append((media, providers))

  def get_state(self):
    return types.FilesManagerState()


def _make_chat_with_image_data():
  """Build a chat with one user message containing one IMAGE block with
  inline data. Suitable for upload (has local content).
  """
  mc = message_content.MessageContent(
      type=message_content.ContentType.IMAGE,
      data=b'\x89PNG\r\n\x1a\n',
      media_type='image/png',
  )
  return chat_session.Chat(messages=[
      message_module.Message(role='user', content=[mc]),
  ]), mc


class TestAutoUploadMedia:

  def test_no_files_manager_noop(self):
    c = _build_connector(_CaptureTestConnector)  # no files_manager
    chat, _ = _make_chat_with_image_data()
    qr = _make_query_record(chat=chat)
    # Does not raise despite having media.
    c._auto_upload_media(qr)

  def test_no_chat_noop(self):
    stub = _StubFilesManager()
    c = _build_connector(_CaptureTestConnector, files_manager=stub)
    qr = _make_query_record(chat=None)
    c._auto_upload_media(qr)
    assert stub.upload_calls == []

  def test_already_uploaded_to_provider_skipped(self):
    stub = _StubFilesManager()
    c = _build_connector(_CaptureTestConnector, files_manager=stub)
    mc = message_content.MessageContent(
        type=message_content.ContentType.IMAGE,
        data=b'\x89PNG\r\n\x1a\n',
        media_type='image/png',
        provider_file_api_ids={'capture_test_provider': 'existing_id'},
    )
    chat = chat_session.Chat(messages=[
        message_module.Message(role='user', content=[mc]),
    ])
    qr = _make_query_record(chat=chat)
    c._auto_upload_media(qr)
    assert stub.upload_calls == []

  def test_unsupported_media_type_skipped(self):
    stub = _StubFilesManager(is_upload_supported_return=False)
    c = _build_connector(_CaptureTestConnector, files_manager=stub)
    chat, _ = _make_chat_with_image_data()
    qr = _make_query_record(chat=chat)
    c._auto_upload_media(qr)
    assert stub.upload_calls == []

  def test_pending_media_uploaded(self):
    stub = _StubFilesManager(is_upload_supported_return=True)
    c = _build_connector(_CaptureTestConnector, files_manager=stub)
    chat, mc = _make_chat_with_image_data()
    qr = _make_query_record(chat=chat)
    c._auto_upload_media(qr)
    assert len(stub.upload_calls) == 1
    uploaded_media, providers = stub.upload_calls[0]
    assert uploaded_media is mc
    assert providers == ['capture_test_provider']


# =============================================================================
# generate() end-to-end orchestration
# =============================================================================


import proxai.connectors.adapter_utils as adapter_utils
import proxai.connectors.providers.mock_provider as mock_provider


def _mock_provider_model():
  return types.ProviderModelType(
      provider='mock_provider', model='mock_model',
      provider_model_identifier='mock_model',
  )


def _mock_provider_model_config():
  return _make_provider_model_config(
      provider='mock_provider', model='mock_model',
      features=types.FeatureConfigType(
          prompt=_S, messages=_S, system_prompt=_S,
          output_format=types.OutputFormatConfigType(text=_S, json=_S),
          parameters=types.ParameterConfigType(
              temperature=_S, max_tokens=_S,
          ),
      ),
      pricing=types.ProviderModelPricingType(
          input_token_cost=10,
          output_token_cost=20,
      ),
  )


def _mock_failing_provider_model():
  return types.ProviderModelType(
      provider='mock_failing_provider', model='mock_failing_model',
      provider_model_identifier='mock_failing_model',
  )


def _mock_failing_provider_model_config():
  return _make_provider_model_config(
      provider='mock_failing_provider', model='mock_failing_model',
      features=types.FeatureConfigType(
          prompt=_S,
          output_format=types.OutputFormatConfigType(text=_S),
      ),
  )


class TestGenerate:

  # Input validation (3) ------------------------------------------------------

  def test_prompt_and_messages_together_raises(self):
    c = _build_connector(mock_provider.MockProviderModelConnector)
    with pytest.raises(ValueError, match='prompt and messages'):
      c.generate(
          provider_model=_mock_provider_model(),
          provider_model_config=_mock_provider_model_config(),
          prompt='hi',
          messages=chat_session.Chat(messages=[
              message_module.Message(role='user', content='hello'),
          ]),
      )

  def test_system_prompt_and_messages_together_raises(self):
    c = _build_connector(mock_provider.MockProviderModelConnector)
    with pytest.raises(ValueError, match='system_prompt and messages'):
      c.generate(
          provider_model=_mock_provider_model(),
          provider_model_config=_mock_provider_model_config(),
          system_prompt='be nice',
          messages=chat_session.Chat(messages=[
              message_module.Message(role='user', content='hello'),
          ]),
      )

  def test_provider_model_mismatch_raises(self):
    c = _build_connector(mock_provider.MockProviderModelConnector)
    wrong_model = types.ProviderModelType(
        provider='wrong_provider', model='m',
        provider_model_identifier='m',
    )
    with pytest.raises(ValueError, match='does not match'):
      c.generate(
          provider_model=wrong_model,
          provider_model_config=_mock_provider_model_config(),
          prompt='hi',
      )

  # Success path (2) ----------------------------------------------------------

  def test_success_runs_executor_and_returns_call_record(self):
    c = _build_connector(mock_provider.MockProviderModelConnector)
    result = c.generate(
        provider_model=_mock_provider_model(),
        provider_model_config=_mock_provider_model_config(),
        prompt='hi',
    )
    assert isinstance(result, types.CallRecord)
    assert result.result.status == types.ResultStatusType.SUCCESS
    assert result.connection.result_source == types.ResultSource.PROVIDER
    assert result.connection.endpoint_used == 'generate.text'
    assert result.result.usage.input_tokens > 0
    assert result.result.timestamp.end_utc_date is not None
    assert result.result.usage.estimated_cost is not None

  def test_keep_raw_provider_response_attaches_debug(self):
    c_keep = _build_connector(
        mock_provider.MockProviderModelConnector,
        keep_raw_provider_response=True,
    )
    r1 = c_keep.generate(
        provider_model=_mock_provider_model(),
        provider_model_config=_mock_provider_model_config(),
        prompt='hi',
    )
    assert r1.debug is not None
    assert r1.debug.raw_provider_response is not None

    c_nokeep = _build_connector(
        mock_provider.MockProviderModelConnector,
        keep_raw_provider_response=False,
    )
    r2 = c_nokeep.generate(
        provider_model=_mock_provider_model(),
        provider_model_config=_mock_provider_model_config(),
        prompt='hi',
    )
    assert r2.debug is None

  # Cache hit path (1) --------------------------------------------------------

  def test_cache_hit_skips_executor_and_sets_result_source_cache(
      self, tmp_path,
  ):
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(
        mock_provider.MockProviderModelConnector,
        query_cache_manager=cache_mgr,
    )
    provider_model = _mock_provider_model()
    output_format = types.OutputFormat(type=types.OutputFormatType.TEXT)
    connection_options = types.ConnectionOptions()

    # Pre-populate cache with a distinctive result.
    qr_for_cache = types.QueryRecord(
        prompt='hi', provider_model=provider_model,
        output_format=output_format, connection_options=connection_options,
    )
    cache_mgr.cache(
        query_record=qr_for_cache,
        result_record=_make_result_record(output_text='FROM_CACHE'),
    )

    result = c.generate(
        provider_model=provider_model,
        provider_model_config=_mock_provider_model_config(),
        prompt='hi',
    )
    assert result.connection.result_source == types.ResultSource.CACHE
    assert result.result.output_text == 'FROM_CACHE'

  def test_cache_miss_preserves_fail_reason_on_returned_record(
      self, tmp_path,
  ):
    """cache_look_fail_reason must survive onto the returned CallRecord."""
    cache_mgr = _make_query_cache_manager(tmp_path)
    c = _build_connector(
        mock_provider.MockProviderModelConnector,
        query_cache_manager=cache_mgr,
    )
    # Cache has no entry for this prompt → CACHE_NOT_FOUND, then the
    # provider runs. The returned record's cache_look_fail_reason must
    # still report CACHE_NOT_FOUND so dashboards can show why the
    # provider was hit.
    result = c.generate(
        provider_model=_mock_provider_model(),
        provider_model_config=_mock_provider_model_config(),
        prompt='never-cached',
    )
    assert result.connection.result_source == types.ResultSource.PROVIDER
    assert (
        result.connection.cache_look_fail_reason
        == types.CacheLookFailReason.CACHE_NOT_FOUND
    )

  # Failure path (2) ----------------------------------------------------------

  def test_failure_with_suppress_false_raises(self):
    c = _build_connector(mock_provider.MockFailingProviderModelConnector)
    with pytest.raises(ValueError, match='Mock failing provider query'):
      c.generate(
          provider_model=_mock_failing_provider_model(),
          provider_model_config=_mock_failing_provider_model_config(),
          prompt='hi',
      )

  def test_failure_with_suppress_false_uploads_stringified_error(self):
    # Regression: on the raise path the call_record must be stringified
    # before upload, otherwise ProxDash ingestion fails with
    # "Object of type <Exception> is not JSON serializable".
    c = _build_connector(mock_provider.MockFailingProviderModelConnector)
    captured = []
    c._upload_call_record_to_proxdash = captured.append
    with pytest.raises(ValueError, match='Mock failing provider query'):
      c.generate(
          provider_model=_mock_failing_provider_model(),
          provider_model_config=_mock_failing_provider_model_config(),
          prompt='hi',
      )
    assert len(captured) == 1
    uploaded = captured[0]
    assert uploaded.result.status == types.ResultStatusType.FAILED
    assert isinstance(uploaded.result.error, str)
    assert 'Mock failing provider query' in uploaded.result.error

  def test_failure_with_suppress_true_returns_stringified_error(self):
    c = _build_connector(mock_provider.MockFailingProviderModelConnector)
    result = c.generate(
        provider_model=_mock_failing_provider_model(),
        provider_model_config=_mock_failing_provider_model_config(),
        prompt='hi',
        connection_options=types.ConnectionOptions(
            suppress_provider_errors=True,
        ),
    )
    assert result.result.status == types.ResultStatusType.FAILED
    assert isinstance(result.result.error, str)
    assert 'Mock failing provider query' in result.result.error


# =============================================================================
# Feature-tag rollup — get_feature_tags_support_level, get_tag_support_level
# =============================================================================


class TestGetFeatureTagsSupportLevel:

  def test_any_endpoint_supported_returns_supported(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _S})
    c = _build_connector(cls)
    # Model also supports prompt so the merge resolves to SUPPORTED.
    result = c.get_feature_tags_support_level(
        feature_tags=[types.FeatureTag.PROMPT],
        model_feature_config=_all_supported_features(),
    )
    assert result == _S

  def test_no_supported_but_best_effort_returns_best_effort(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _BE})
    c = _build_connector(cls)
    result = c.get_feature_tags_support_level(
        feature_tags=[types.FeatureTag.PROMPT],
        model_feature_config=_all_supported_features(),
    )
    assert result == _BE


class TestGetTagSupportLevel:

  def test_generic_tag_rollup_supported(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _S})
    c = _build_connector(cls)
    result = c.get_tag_support_level(
        tags=[types.FeatureTag.PROMPT],
        resolve_fn=adapter_utils.resolve_feature_tag_support,
        model_feature_config=_all_supported_features(),
    )
    assert result == _S

  def test_generic_tag_rollup_best_effort(self):
    cls = _make_connector_with_endpoints({'first': _NS, 'second': _BE})
    c = _build_connector(cls)
    result = c.get_tag_support_level(
        tags=[types.FeatureTag.PROMPT],
        resolve_fn=adapter_utils.resolve_feature_tag_support,
        model_feature_config=_all_supported_features(),
    )
    assert result == _BE


# =============================================================================
# Multi-choice response shape (n > 1)
# =============================================================================
# Contract (see call_record_analysis §2.9):
#   content  = [first provider choice]          (always when the call succeeded)
#   choices  = [ChoiceType for each remaining]  (length n-1) when n > 1
#            = None                             when n == 1
# These tests pin the contract per-connector so a regression in one
# provider is caught immediately.


class _FakeMessage:

  def __init__(self, content):
    self.content = content
    self.parsed = None


class _FakeChoice:

  def __init__(self, content):
    self.message = _FakeMessage(content)


class _FakeResponse:
  """Minimal stand-in for an SDK chat-completions response."""

  def __init__(self, choice_texts):
    self.choices = [_FakeChoice(t) for t in choice_texts]


def _install_fake_chat_completions(connector, choice_texts):
  """Replace connector.api.chat.completions.create with a stub."""
  response = _FakeResponse(choice_texts)

  def _create(*args, **kwargs):
    return response

  connector.api.chat.completions.create = _create


def _install_fake_mistral_chat(connector, choice_texts):
  """Replace connector.api.chat.complete with a stub (Mistral's surface)."""
  response = _FakeResponse(choice_texts)

  def _complete(*args, **kwargs):
    return response

  connector.api.chat.complete = _complete


class TestMultiChoiceShape:

  def _provider_model(self, provider, model):
    return types.ProviderModelType(
        provider=provider, model=model, provider_model_identifier=model,
    )

  def _qr(self, provider, model, n=3):
    return types.QueryRecord(
        prompt='hi',
        provider_model=self._provider_model(provider, model),
        parameters=types.ParameterType(n=n),
        output_format=types.OutputFormat(type=types.OutputFormatType.TEXT),
    )

  # ---- OpenAI -------------------------------------------------------------
  def test_openai_n3_content_is_first_choices_is_remainder(self):
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    _install_fake_chat_completions(c, ['A', 'B', 'C'])

    result = c._chat_completions_create_executor(
        query_record=self._qr('openai', 'gpt-4o', n=3),
    )
    assert result.result_record.error is None
    assert result.result_record.content[0].text == 'A'
    assert len(result.result_record.choices) == 2
    assert result.result_record.choices[0].content[0].text == 'B'
    assert result.result_record.choices[1].content[0].text == 'C'

  def test_openai_n1_leaves_choices_none(self):
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    _install_fake_chat_completions(c, ['only'])

    result = c._chat_completions_create_executor(
        query_record=self._qr('openai', 'gpt-4o', n=1),
    )
    assert result.result_record.content[0].text == 'only'
    assert result.result_record.choices is None

  # ---- Mistral ------------------------------------------------------------
  def test_mistral_n3_content_is_first_choices_is_remainder(self):
    from proxai.connectors.providers import mistral as mistral_connector
    c = _build_connector(mistral_connector.MistralConnector)
    _install_fake_mistral_chat(c, ['A', 'B', 'C'])

    result = c._chat_complete_executor(
        query_record=self._qr('mistral', 'mistral-large-latest', n=3),
    )
    assert result.result_record.error is None
    assert result.result_record.content[0].text == 'A'
    assert len(result.result_record.choices) == 2
    assert result.result_record.choices[0].content[0].text == 'B'
    assert result.result_record.choices[1].content[0].text == 'C'

  def test_mistral_n1_leaves_choices_none(self):
    from proxai.connectors.providers import mistral as mistral_connector
    c = _build_connector(mistral_connector.MistralConnector)
    _install_fake_mistral_chat(c, ['only'])

    result = c._chat_complete_executor(
        query_record=self._qr('mistral', 'mistral-large-latest', n=1),
    )
    assert result.result_record.content[0].text == 'only'
    assert result.result_record.choices is None

  # ---- Databricks ---------------------------------------------------------
  def test_databricks_n3_content_is_first_choices_is_remainder(self):
    from proxai.connectors.providers import databricks as databricks_connector
    c = _build_connector(databricks_connector.DatabricksConnector)
    _install_fake_chat_completions(c, ['A', 'B', 'C'])

    result = c._chat_completions_create_executor(
        query_record=self._qr(
            'databricks', 'databricks-dbrx-instruct', n=3,
        ),
    )
    assert result.result_record.error is None
    # databricks._parse_message_content returns list[MessageContent]; TEXT
    # for plain strings.
    assert result.result_record.content[0].text == 'A'
    assert len(result.result_record.choices) == 2
    assert result.result_record.choices[0].content[0].text == 'B'
    assert result.result_record.choices[1].content[0].text == 'C'

  def test_databricks_n1_leaves_choices_none(self):
    from proxai.connectors.providers import databricks as databricks_connector
    c = _build_connector(databricks_connector.DatabricksConnector)
    _install_fake_chat_completions(c, ['only'])

    result = c._chat_completions_create_executor(
        query_record=self._qr(
            'databricks', 'databricks-dbrx-instruct', n=1,
        ),
    )
    assert result.result_record.content[0].text == 'only'
    assert result.result_record.choices is None


# =============================================================================
# OpenAI responses.create: web_search + JSON/PYDANTIC output conflict
# =============================================================================


class TestOpenAIResponsesCreateNativeJsonModeBlocked:
  """OpenAI's responses.create rejects 'text.format=json_object' alongside
  web_search. The executor must skip setting native JSON mode when the
  query pairs WEB_SEARCH with a structured output format; FeatureAdapter
  has already injected the schema/json guidance into the prompt, so the
  model still gets steered toward structured output.
  """

  def _provider_model(self):
    return types.ProviderModelType(
        provider='openai', model='gpt-4o',
        provider_model_identifier='gpt-4o',
    )

  def _install_capturing_responses_create(self, connector):
    captured = {}

    class _FakeResponsesResponse:

      def __init__(self):
        self.output = []

    def _create(**kwargs):
      captured.update(kwargs)
      return _FakeResponsesResponse()

    connector.api.responses.create = _create
    return captured

  def test_predicate_true_when_web_search_tool_present(self):
    from proxai.connectors.providers import openai as openai_connector
    qr = types.QueryRecord(tools=[types.Tools.WEB_SEARCH])
    assert openai_connector.OpenAIConnector._native_json_mode_blocked(qr)

  def test_predicate_false_when_no_tools(self):
    from proxai.connectors.providers import openai as openai_connector
    qr = types.QueryRecord(tools=None)
    assert not openai_connector.OpenAIConnector._native_json_mode_blocked(qr)

  def test_executor_drops_native_json_when_web_search_requested(self):
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    captured = self._install_capturing_responses_create(c)

    c._responses_create_executor(
        query_record=types.QueryRecord(
            prompt='hi',
            provider_model=self._provider_model(),
            tools=[types.Tools.WEB_SEARCH],
            output_format=types.OutputFormat(
                type=types.OutputFormatType.JSON),
        ),
    )
    assert captured.get('tools') == [{'type': 'web_search'}]
    assert 'text' not in captured

  def test_executor_keeps_native_json_without_web_search(self):
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    captured = self._install_capturing_responses_create(c)

    c._responses_create_executor(
        query_record=types.QueryRecord(
            prompt='hi',
            provider_model=self._provider_model(),
            output_format=types.OutputFormat(
                type=types.OutputFormatType.JSON),
        ),
    )
    assert captured.get('text') == {'format': {'type': 'json_object'}}
    assert 'tools' not in captured

  def _install_scripted_responses_create(self, connector, text):
    """Replace responses.create to return a single text content block."""

    class _FakeContent:

      def __init__(self, text):
        self.text = text
        self.annotations = None

    class _FakeOutput:

      def __init__(self, content_text):
        self.type = 'message'
        self.content = [_FakeContent(content_text)]

    class _FakeResponse:

      def __init__(self, content_text):
        self.output = [_FakeOutput(content_text)]

    def _create(**kwargs):
      return _FakeResponse(text)

    connector.api.responses.create = _create

  def test_executor_unwraps_fenced_json_into_json_block(self):
    # Regression: when native JSON mode is blocked by web_search, the model
    # returns markdown-fenced JSON text. The executor must extract the JSON
    # so ResultAdapter sees a clean JSON block (contract: output_format JSON
    # => content is a JSON block, never TEXT).
    import proxai.chat.message_content as message_content
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    self._install_scripted_responses_create(
        c, '```json\n{"a": 1, "b": "two"}\n```')

    result = c._responses_create_executor(
        query_record=types.QueryRecord(
            prompt='hi',
            provider_model=self._provider_model(),
            tools=[types.Tools.WEB_SEARCH],
            output_format=types.OutputFormat(
                type=types.OutputFormatType.JSON),
        ),
    )
    content = result.result_record.content
    assert len(content) == 1
    assert content[0].type == message_content.ContentType.JSON
    assert content[0].json == {'a': 1, 'b': 'two'}

  def test_executor_wraps_clean_json_text_into_json_block(self):
    # The native JSON mode path also emits a JSON block (not TEXT), so the
    # contract with ResultAdapter is uniform across degraded and native.
    import proxai.chat.message_content as message_content
    from proxai.connectors.providers import openai as openai_connector
    c = _build_connector(openai_connector.OpenAIConnector)
    self._install_scripted_responses_create(c, '{"a": 1, "b": "two"}')

    result = c._responses_create_executor(
        query_record=types.QueryRecord(
            prompt='hi',
            provider_model=self._provider_model(),
            output_format=types.OutputFormat(
                type=types.OutputFormatType.JSON),
        ),
    )
    content = result.result_record.content
    assert len(content) == 1
    assert content[0].type == message_content.ContentType.JSON
    assert content[0].json == {'a': 1, 'b': 'two'}


class TestClose:
  """Lifecycle: close() releases the SDK client; client stays usable."""

  def test_close_calls_api_close_and_clears_attribute(self):
    c = _build_connector(_CaptureTestConnector)
    fake_api = mock.MagicMock()
    c._api = fake_api
    c.close()
    fake_api.close.assert_called_once_with()
    assert c._api is None

  def test_close_when_api_never_initialized_is_noop(self):
    c = _build_connector(_CaptureTestConnector)
    assert getattr(c, '_api', None) is None
    c.close()  # must not raise.
    assert getattr(c, '_api', None) is None

  def test_close_idempotent(self):
    c = _build_connector(_CaptureTestConnector)
    c._api = mock.MagicMock()
    c.close()
    c.close()  # second call must not raise.

  def test_close_tolerates_missing_close_method(self):
    c = _build_connector(_CaptureTestConnector)
    c._api = object()  # plain object — no .close attribute.
    c.close()  # must not raise.
    assert c._api is None

  def test_close_swallows_close_exception(self):
    c = _build_connector(_CaptureTestConnector)
    fake_api = mock.MagicMock()
    fake_api.close.side_effect = RuntimeError('boom')
    c._api = fake_api
    c.close()  # must not propagate.
    assert c._api is None

  def test_context_manager_calls_close_on_exit(self):
    c = _build_connector(_CaptureTestConnector)
    fake_api = mock.MagicMock()
    c._api = fake_api
    with c as ctx:
      assert ctx is c
    fake_api.close.assert_called_once_with()
    assert c._api is None

  def test_close_then_reuse_relazy_inits(self):
    # init_mock_model returns None for _CaptureTestConnector, so we patch
    # it to fresh sentinels and verify close → next access triggers
    # another build.
    c = _build_connector(_CaptureTestConnector)
    sentinels = [mock.MagicMock(), mock.MagicMock()]
    with mock.patch.object(
        c, 'init_mock_model', side_effect=sentinels) as init_spy:
      _ = c.api  # first lazy init.
      assert init_spy.call_count == 1
      c.close()
      _ = c.api  # second lazy init after close.
      assert init_spy.call_count == 2

"""Focused tests for the keep_raw_provider_response capture point.

These tests construct a minimal ProviderConnector subclass and exercise
_safe_provider_query(), _execute_call(), and the ExecutorResult return
contract directly.
"""
import proxai.connectors.provider_connector as provider_connector
import proxai.types as types

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


def _build_connector(connector_cls, *, keep_raw_provider_response):
  params = provider_connector.ProviderConnectorParams(
      run_type=types.RunType.TEST,
      feature_mapping_strategy=types.FeatureMappingStrategy.BEST_EFFORT,
      logging_options=types.LoggingOptions(),
      provider_token_value_map={'CAPTURE_TEST_KEY': 'unused'},
      keep_raw_provider_response=keep_raw_provider_response,
  )
  return connector_cls(init_from_params=params)


class TestExecutorResultContract:

  def test_success_carries_raw_response(self):
    connector = _build_connector(
        _CaptureTestConnector, keep_raw_provider_response=True
    )
    query_record = types.QueryRecord(
        prompt='hi',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.TEXT
        ),
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=query_record,
        provider_model_config=_provider_model_config(),
    )
    assert isinstance(executor_result, types.ExecutorResult)
    assert executor_result.raw_provider_response is _RAW_SUCCESS_SENTINEL
    assert executor_result.result_record.status == types.ResultStatusType.SUCCESS

  def test_success_without_flag_still_returns_raw(self):
    connector = _build_connector(
        _CaptureTestConnector, keep_raw_provider_response=False
    )
    query_record = types.QueryRecord(
        prompt='hi',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.TEXT
        ),
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=query_record,
        provider_model_config=_provider_model_config(),
    )
    assert executor_result.raw_provider_response is _RAW_SUCCESS_SENTINEL
    assert executor_result.result_record.status == types.ResultStatusType.SUCCESS

  def test_failure_returns_none_raw_response(self):
    connector = _build_connector(
        _FailingCaptureTestConnector, keep_raw_provider_response=True
    )
    query_record = types.QueryRecord(
        prompt='hi',
        output_format=types.OutputFormat(
            type=types.OutputFormatType.TEXT
        ),
    )
    executor_result = connector._execute_call(
        chosen_executor=connector._generate_text_executor,
        chosen_endpoint='generate.text',
        query_record=query_record,
        provider_model_config=_provider_model_config(),
    )
    assert executor_result.raw_provider_response is None
    assert executor_result.result_record.status == types.ResultStatusType.FAILED


def _provider_model_config():
  return types.ProviderModelConfig(
      provider_model=types.ProviderModelType(
          provider='capture_test_provider',
          model='capture_test_model',
          provider_model_identifier='capture_test_model',
      ),
      pricing=None,
      features=types.FeatureConfigType(),
      metadata=None,
  )

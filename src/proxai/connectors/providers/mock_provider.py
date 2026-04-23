import time
from typing import Any

import pydantic

import proxai.connectors.provider_connector as provider_connector
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
OutputFormatConfigType = types.OutputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType


class SamplePydanticModel(pydantic.BaseModel):
  """Sample Pydantic model for mock testing."""

  name: str
  age: int


class MockProviderModelConnector(provider_connector.ProviderConnector):
  """Mock connector for testing without real API calls."""

  PROVIDER_NAME = 'mock_provider'

  PROVIDER_API_KEYS = []

  ENDPOINT_PRIORITY = [
      'generate.text',
  ]

  ENDPOINT_CONFIG = {
      'generate.text': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.BEST_EFFORT,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  ENDPOINT_EXECUTORS = {
      'generate.text': '_generate_text_executor',
  }

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _generate_text_executor(
      self, query_record: types.QueryRecord) -> types.ExecutorResult:
    def _mock_provider_query():
      return 'mock response'

    response, result_record = self._safe_provider_query(_mock_provider_query)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    fmt = query_record.output_format
    is_text = fmt is None or fmt.type == types.OutputFormatType.TEXT
    is_json = fmt and fmt.type == types.OutputFormatType.JSON
    is_pydantic = fmt and fmt.type == types.OutputFormatType.PYDANTIC
    if is_text:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text='mock response',
          )
      ]
    elif is_json:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.JSON,
              json={"name": "John Doe", "age": 30},
          )
      ]
    elif is_pydantic:
      # Respect the requested pydantic_class. For the hard-coded
      # SamplePydanticModel we know the shape; for any other class the
      # mock uses `model_construct()` to emit a valid instance without
      # having to synthesize field values — good enough for framework
      # plumbing tests (probe dispatch, STRICT-vs-BEST_EFFORT, etc.).
      pydantic_class = fmt.pydantic_class
      if pydantic_class is SamplePydanticModel:
        instance_value = SamplePydanticModel(name='John Doe', age=30)
      else:
        instance_value = pydantic_class.model_construct()
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name=pydantic_class.__name__,
                  class_value=pydantic_class,
                  instance_value=instance_value,
              ),
          )
      ]

    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)


class MockFailingProviderModelConnector(provider_connector.ProviderConnector):
  """Mock connector that always fails for testing error handling."""

  PROVIDER_NAME = 'mock_failing_provider'

  PROVIDER_API_KEYS = []

  ENDPOINT_PRIORITY = [
      'generate.text',
  ]

  ENDPOINT_CONFIG = {
      'generate.text': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.BEST_EFFORT,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  ENDPOINT_EXECUTORS = {
      'generate.text': '_generate_text_executor',
  }

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _generate_text_executor(
      self, query_record: types.QueryRecord) -> types.ExecutorResult:
    def _failing_provider_query():
      raise ValueError('Mock failing provider query')
    response, result_record = self._safe_provider_query(_failing_provider_query)
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)


class MockSlowProviderModelConnector(provider_connector.ProviderConnector):
  """Mock connector with delayed responses for testing timeouts."""

  PROVIDER_NAME = 'mock_slow_provider'

  PROVIDER_API_KEYS = []

  ENDPOINT_PRIORITY = [
      'generate.text',
  ]

  ENDPOINT_CONFIG = {
      'generate.text': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.BEST_EFFORT,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  ENDPOINT_EXECUTORS = {
      'generate.text': '_generate_text_executor',
  }

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def _generate_text_executor(
      self, query_record: types.QueryRecord) -> types.ExecutorResult:
    def _slow_provider_query():
      time.sleep(120)
      return 'mock response'
    response, result_record = self._safe_provider_query(_slow_provider_query)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    fmt = query_record.output_format
    is_text = fmt is None or fmt.type == types.OutputFormatType.TEXT
    is_json = (
        fmt and (
            fmt.type == types.OutputFormatType.JSON or
            fmt.type == types.OutputFormatType.PYDANTIC
        )
    )
    if is_text:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text='mock response',
          )
      ]
    elif is_json:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text='{"name": "John Doe", "age": 30}',
          )
      ]
    elif fmt and fmt.type == types.OutputFormatType.PYDANTIC:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name='SamplePydanticModel',
                  class_value=SamplePydanticModel,
                  instance_value=SamplePydanticModel(
                      name='John Doe', age=30),
              ),
          )
      ]

    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

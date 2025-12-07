import time
import pydantic
import proxai.types as types
import proxai.connectors.model_connector as model_connector


class SamplePydanticModel(pydantic.BaseModel):
  name: str
  age: int


class MockProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    if query_record.response_format is None:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name='SamplePydanticModel',
              instance_value=SamplePydanticModel(
                name='John Doe',
                age=30)),
          type=types.ResponseType.PYDANTIC)


class MockFailingProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_failing_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(self, query_record: types.QueryRecord):
    raise ValueError('Temp Error')


class MockSlowProviderModelConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return "mock_slow_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    time.sleep(120)

    if query_record.response_format is None:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name='SamplePydanticModel',
              instance_value=SamplePydanticModel(
                name='John Doe',
                age=30)),
          type=types.ResponseType.PYDANTIC)

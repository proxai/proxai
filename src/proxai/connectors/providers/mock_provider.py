import time
import functools
from typing import Any, Callable, Optional
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

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    return functools.partial(query_function, system=system_message)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(query_function, response_format='json')

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(query_function, response_format='json_schema')

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(query_function, response_format='pydantic')

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return str(response)

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return response if isinstance(response, dict) else {"value": response}

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return response if isinstance(response, dict) else {"value": response}

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord):
    return response

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

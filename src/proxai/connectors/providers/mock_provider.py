import functools
import time
from collections.abc import Callable
from typing import Any

import pydantic

import proxai.connectors.model_connector as model_connector
import proxai.types as types


class SamplePydanticModel(pydantic.BaseModel):
  """Sample Pydantic model for mock testing."""

  name: str
  age: int


class MockProviderModelConnector(model_connector.ProviderModelConnector):
  """Mock connector for testing without real API calls."""

  def get_provider_name(self):
    return "mock_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: str | None = None) -> Callable:
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
    fmt = query_record.response_format
    is_text = fmt is None or fmt.type == types.ResponseFormatType.TEXT
    is_json = (fmt and (fmt.type == types.ResponseFormatType.JSON or
               fmt.type == types.ResponseFormatType.JSON_SCHEMA))
    if is_text:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif is_json:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif fmt and fmt.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=SamplePydanticModel(name='John Doe', age=30),
          type=types.ResponseType.PYDANTIC,
          pydantic_metadata=types.PydanticMetadataType(
              class_name='SamplePydanticModel'))


class MockFailingProviderModelConnector(model_connector.ProviderModelConnector):
  """Mock connector that always fails for testing error handling."""

  def get_provider_name(self):
    return "mock_failing_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(self, query_record: types.QueryRecord):
    raise ValueError('Temp Error')


class MockSlowProviderModelConnector(model_connector.ProviderModelConnector):
  """Mock connector with delayed responses for testing timeouts."""

  def get_provider_name(self):
    return "mock_slow_provider"

  def init_model(self):
    return None

  def init_mock_model(self):
    return None

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    time.sleep(120)

    fmt = query_record.response_format
    is_text = fmt is None or fmt.type == types.ResponseFormatType.TEXT
    is_json = (fmt and (fmt.type == types.ResponseFormatType.JSON or
               fmt.type == types.ResponseFormatType.JSON_SCHEMA))
    if is_text:
      return types.Response(
          value="mock response",
          type=types.ResponseType.TEXT)
    elif is_json:
      return types.Response(
          value={"name": "John Doe", "age": 30},
          type=types.ResponseType.JSON)
    elif fmt and fmt.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=SamplePydanticModel(name='John Doe', age=30),
          type=types.ResponseType.PYDANTIC,
          pydantic_metadata=types.PydanticMetadataType(
              class_name='SamplePydanticModel'))

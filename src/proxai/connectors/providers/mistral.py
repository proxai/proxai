from typing import Any, Callable
import functools
import json
import os
import re
from mistralai import Mistral
from mistralai.models import ResponseFormat, JSONSchema
import proxai.types as types
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.connectors.model_connector as model_connector


class MistralConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'mistral'

  def init_model(self):
    return Mistral(api_key=os.environ.get('MISTRAL_API_KEY'))

  def init_mock_model(self):
    return mistral_mock.MistralMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      # Use chat.parse for Pydantic models
      return functools.partial(self.api.chat.parse)
    else:
      # Use chat.complete for other formats
      return functools.partial(self.api.chat.complete)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system is not None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'user':
          query_messages.append({'role': 'user', 'content': message['content']})
        if message['role'] == 'assistant':
          query_messages.append({'role': 'assistant', 'content': message['content']})
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      create = functools.partial(create, stop=query_record.stop)

    # Handle response format configuration (for non-Pydantic formats)
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        create = functools.partial(
            create,
            response_format=ResponseFormat(type='json_object'))
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        schema_name = json_schema_obj.get('name', 'response_schema')
        raw_schema = json_schema_obj.get('schema', json_schema_obj)
        json_schema = JSONSchema(
            name=schema_name,
            schema=raw_schema)
        create = functools.partial(
            create,
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=json_schema))
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        create = functools.partial(
            create,
            response_format=query_record.response_format.value.class_value)

    return create

  def _response_mapping(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if query_record.response_format is None:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=self._extract_json_from_text(
              response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=self._extract_json_from_text(
              response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=response.choices[0].message.parsed),
          type=types.ResponseType.PYDANTIC)

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

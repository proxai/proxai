from typing import Any, Callable, Optional
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
        query_record.response_format.type ==
        types.ResponseFormatType.PYDANTIC and
        'response_format::pydantic' in
        self.provider_model_config.features.supported):
      # Use chat.parse for Pydantic models
      return functools.partial(self.api.chat.parse)
    else:
      # Use chat.complete for other formats
      return functools.partial(self.api.chat.complete)

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    if system_message is None:
      return query_function
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.insert(0, {'role': 'system', 'content': system_message})
    return functools.partial(query_function, messages=messages)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        response_format=ResponseFormat(type='json_object'))

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    schema_value = query_record.response_format.value
    json_schema_obj = schema_value['json_schema']
    schema_name = json_schema_obj.get('name', 'response_schema')
    raw_schema = json_schema_obj.get('schema', json_schema_obj)
    json_schema = JSONSchema(
        name=schema_name,
        schema=raw_schema)
    return functools.partial(
        query_function,
        response_format=ResponseFormat(
            type='json_schema',
            json_schema=json_schema))

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        response_format=query_record.response_format.value.class_value)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
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

    create = self.add_system_and_response_format_params(create, query_record)

    return create

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return response.choices[0].message.content

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    return response.choices[0].message.parsed

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

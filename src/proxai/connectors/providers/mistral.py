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
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.complete':
      return functools.partial(self.api.chat.complete)
    elif chosen_endpoint == 'chat.parse':
      return functools.partial(self.api.chat.parse)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        messages=[{'role': 'user', 'content': query_record.prompt}])

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    converted_messages = []
    for message in query_record.messages:
      if message['role'] == 'user':
        converted_messages.append(
            {'role': 'user', 'content': message['content']})
      elif message['role'] == 'assistant':
        converted_messages.append(
            {'role': 'assistant', 'content': message['content']})

    messages = query_function.keywords.get('messages')
    if messages is None:
      return functools.partial(
          query_function,
          messages=converted_messages)
    else:
      messages = converted_messages + messages
      return functools.partial(
          query_function,
          messages=messages)

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.insert(0, {'role': 'system', 'content': query_record.system})
    return functools.partial(query_function, messages=messages)

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        max_tokens=query_record.max_tokens)

  def temperature_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        temperature=query_record.temperature)

  def stop_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        stop=query_record.stop)

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
    return functools.partial(
        query_function,
        response_format=query_record.response_format.value)

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint == 'chat.complete':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.complete. Code should never reach here.')
    elif query_record.chosen_endpoint == 'chat.parse':
      return functools.partial(
          query_function,
          response_format=query_record.response_format.value.class_value)

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'Web search is not supported for Mistral. Code should never reach here.')

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
    if query_record.chosen_endpoint == 'chat.complete':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.complete. Code should never reach here.')
    elif query_record.chosen_endpoint == 'chat.parse':
      return response.choices[0].message.parsed

  def generate_text_proc(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record.chosen_endpoint)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self.add_features_to_query_function(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

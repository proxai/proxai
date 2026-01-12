import functools
import os
from collections.abc import Callable
from typing import Any

from openai import OpenAI

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types


class DeepSeekConnector(model_connector.ProviderModelConnector):
  """Connector for DeepSeek models."""

  def get_provider_name(self):
    return 'deepseek'

  def init_model(self):
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com")

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.completions.create':
      return functools.partial(self.api.chat.completions.create)
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
    # Note: DeepSeek uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    messages = query_function.keywords.get('messages')
    if messages is None:
      return functools.partial(
          query_function,
          messages=query_record.messages)
    else:
      messages = query_record.messages + messages
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
        max_completion_tokens=query_record.max_tokens)

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

  def _add_json_guidance_to_user_message(
      self,
      query_function: Callable):
    # NOTE: DeepSeek API expects the JSON guidance to be in the user message.
    # This is a workaround to add JSON guidance to the user message.
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding JSON guidance.')
    for message in messages:
      if message['role'] == 'user':
        if 'json' not in message['content'].lower():
          message['content'] = (
              f'{message["content"]}\n\nYou must respond with valid JSON.')
        break
    return functools.partial(query_function, messages=messages)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    query_function = self._add_json_guidance_to_user_message(query_function)
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    query_function = self._add_json_guidance_to_user_message(query_function)
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    # Note: DeepSeek doesn't have native pydantic support.
    # We use json_object and parse manually in format_pydantic_response.
    query_function = self._add_json_guidance_to_user_message(query_function)
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'Web search is not supported for DeepSeek.')

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
    pydantic_class = query_record.response_format.value.class_value
    return pydantic_class.model_validate_json(
        response.choices[0].message.content)

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

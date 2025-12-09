import functools
import json
from typing import Any, Callable, Optional
from openai import OpenAI
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class OpenAIConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'openai'

  def init_model(self):
    return OpenAI()

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      return functools.partial(self.api.beta.chat.completions.parse)
    else:
      return functools.partial(self.api.chat.completions.create)

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
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    # NOTE: The weird OpenAI API expects the JSON to be in the user message.
    # TODO: Find a better way to do this.
    for message in messages:
      if message['role'] == 'user':
        if 'json' not in message['content']:
          message['content'] = (
              f'{message["content"]}\n\nYou must respond with valid JSON.')
        break
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

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

    # Note: OpenAI uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(
          create, max_completion_tokens=query_record.max_tokens)
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

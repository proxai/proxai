import functools
from collections.abc import Callable
from typing import Any

from openai import OpenAI

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types


class OpenAIConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'openai'

  def init_model(self):
    return OpenAI()

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.completions.create':
      return functools.partial(self.api.chat.completions.create)
    elif chosen_endpoint == 'beta.chat.completions.parse':
      return functools.partial(self.api.beta.chat.completions.parse)
    elif chosen_endpoint == 'responses.create':
      return functools.partial(self.api.responses.create)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
      return functools.partial(
          query_function,
          messages=[{'role': 'user', 'content': query_record.prompt}])
    elif query_record.chosen_endpoint == 'responses.create':
      return functools.partial(
          query_function,
          input=query_record.prompt)

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
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
    elif query_record.chosen_endpoint == 'responses.create':
      raise Exception(
          'Responses.create does not support messages parameter. Code should '
          'never reach here.')

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
      messages = query_function.keywords.get('messages')
      if messages is None:
        raise Exception('Set messages parameter before adding system message.')
      messages.insert(0, {'role': 'system', 'content': query_record.system})
      return functools.partial(query_function, messages=messages)
    elif query_record.chosen_endpoint == 'responses.create':
      return functools.partial(
          query_function,
          instructions=query_record.system)

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
      return functools.partial(
          query_function,
          max_completion_tokens=query_record.max_tokens)
    elif query_record.chosen_endpoint == 'responses.create':
      return functools.partial(
          query_function,
          max_output_tokens=query_record.max_tokens)

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
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
      return functools.partial(
          query_function,
          stop=query_record.stop)
    elif query_record.chosen_endpoint == 'responses.create':
      raise Exception(
          'Responses.create does not support stop parameter. Code should '
          'never reach here.')

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint == 'chat.completions.create':
      return functools.partial(
          query_function,
          response_format={'type': 'json_object'})
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      raise Exception(
          'JSON response format is not supported for '
          'beta.chat.completions.parse. Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      return functools.partial(
          query_function,
          text={'format': {'type': 'json_object'}})

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint == 'chat.completions.create':
      return functools.partial(
          query_function,
          response_format=query_record.response_format.value)
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      raise Exception(
          'JSON schema response format is not supported for '
          'beta.chat.completions.parse. Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      raise Exception(
          'JSON schema response format is not supported for '
          'responses.create yet in ProxAI. It uses different json_schema '
          'format and requires a different mapping. Code should never reach '
          'here. Please reach out to ProxAI team if you need this feature.\n'
          'GitHub: https://github.com/proxai/')

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if query_record.chosen_endpoint == 'chat.completions.create':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.completions.create. Code should never reach here.')
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      return functools.partial(
          query_function,
          response_format=query_record.response_format.value.class_value)
    elif query_record.chosen_endpoint == 'responses.create':
      raise Exception(
          'Pydantic response format is not supported for '
          'responses.create. Code should never reach here.')

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    if (query_record.chosen_endpoint == 'chat.completions.create' or
        query_record.chosen_endpoint == 'beta.chat.completions.parse'):
      raise Exception(
          'Web search is not supported for '
          'chat.completions.create or beta.chat.completions.parse. '
          'Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      return functools.partial(
          query_function,
          tools=[{"type": "web_search"}])

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    if query_record.chosen_endpoint == 'chat.completions.create':
      return response.choices[0].message.content
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      raise Exception(
          'Text response format is not supported for '
          'beta.chat.completions.parse. Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      return response.output_text

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    if query_record.chosen_endpoint == 'chat.completions.create':
      return self._extract_json_from_text(
          response.choices[0].message.content)
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      raise Exception(
          'JSON response format is not supported for '
          'beta.chat.completions.parse. Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      return self._extract_json_from_text(response.output_text)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    if query_record.chosen_endpoint == 'chat.completions.create':
      return self._extract_json_from_text(
          response.choices[0].message.content)
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      raise Exception(
          'JSON schema response format is not supported for '
          'beta.chat.completions.parse. Code should never reach here.')
    elif query_record.chosen_endpoint == 'responses.create':
      return self._extract_json_from_text(response.output_text)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    if query_record.chosen_endpoint == 'chat.completions.create':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.completions.create. Code should never reach here.')
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      return response.choices[0].message.parsed
    elif query_record.chosen_endpoint == 'responses.create':
      return response.output_parsed

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

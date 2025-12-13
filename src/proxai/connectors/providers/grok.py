from typing import Any, Callable, Optional
import functools
import json
import os
from openai import OpenAI
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class GrokConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'grok'

  def init_model(self):
    return OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1")

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.completions.create':
      return functools.partial(self.api.chat.completions.create)
    elif chosen_endpoint == 'beta.chat.completions.parse':
      return functools.partial(self.api.beta.chat.completions.parse)
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
    # Note: There is a bug in the grok api for grok-3-mini-beta and
    # grok-3-mini-fast-beta that if max_completion_tokens is not set,
    # the response is empty string.
    # TODO: Remove this workaround once the bug is fixed.
    if query_record.max_tokens is not None:
      return functools.partial(
          query_function,
          max_completion_tokens=query_record.max_tokens)
    elif self.provider_model.model in [
        'grok-3-mini-beta', 'grok-3-mini-fast-beta']:
      if (query_record.feature_mapping_strategy ==
          types.FeatureMappingStrategy.BEST_EFFORT):
        return functools.partial(
            query_function,
            max_completion_tokens=1000000)
    return query_function

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
    if isinstance(query_record.stop, str):
      return functools.partial(
          query_function,
          stop=[query_record.stop])
    else:
      return functools.partial(
          query_function,
          stop=query_record.stop)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
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
    if query_record.chosen_endpoint == 'chat.completions.create':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.completions.create. Code should never reach here.')
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
      return functools.partial(
          query_function,
          response_format=query_record.response_format.value.class_value)

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'Web search is not supported for Grok. Code should never reach here.')

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
    if query_record.chosen_endpoint == 'chat.completions.create':
      raise Exception(
          'Pydantic response format is not supported for '
          'chat.completions.create. Code should never reach here.')
    elif query_record.chosen_endpoint == 'beta.chat.completions.parse':
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

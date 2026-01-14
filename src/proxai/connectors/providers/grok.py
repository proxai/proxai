import functools
from collections.abc import Callable
from typing import Any

from xai_sdk import Client
from xai_sdk.chat import assistant, system, user
from xai_sdk.tools import web_search

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.grok_mock as grok_mock
import proxai.types as types


class GrokConnector(model_connector.ProviderModelConnector):
  """Connector for xAI Grok models."""

  def get_provider_name(self):
    return 'grok'

  def get_required_provider_token_names(self) -> list[str]:
    return ['XAI_API_KEY']

  def init_model(self):
    return Client(api_key=self.provider_token_value_map['XAI_API_KEY'])

  def init_mock_model(self):
    return grok_mock.GrokMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.create':
      return functools.partial(self.api.chat.create)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_after_construction = query_function.keywords.get(
        'params_after_construction', {})
    messages = params_after_construction.get('messages', [])
    messages.append(user(query_record.prompt))
    params_after_construction['messages'] = messages
    return functools.partial(
        query_function,
        params_after_construction=params_after_construction)

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_after_construction = query_function.keywords.get(
        'params_after_construction', {})
    existing_messages = params_after_construction.get('messages', [])

    messages = []
    for message in query_record.messages:
      if message['role'] == 'user':
        messages.append(user(message['content']))
      elif message['role'] == 'assistant':
        messages.append(assistant(message['content']))
      elif message['role'] == 'system':
        messages.append(system(message['content']))

    params_after_construction['messages'] = existing_messages + messages
    return functools.partial(
        query_function,
        params_after_construction=params_after_construction)

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_after_construction = query_function.keywords.get(
        'params_after_construction', {})
    messages = params_after_construction.get('messages', None)
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.append(system(query_record.system))
    params_after_construction['messages'] = messages
    return functools.partial(
        query_function,
        params_after_construction=params_after_construction)

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_at_construction = query_function.keywords.get(
        'params_at_construction', {})
    params_at_construction['max_tokens'] = query_record.max_tokens
    return functools.partial(
        query_function,
        params_at_construction=params_at_construction)

  def temperature_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_at_construction = query_function.keywords.get(
        'params_at_construction', {})
    params_at_construction['temperature'] = query_record.temperature
    return functools.partial(
        query_function,
        params_at_construction=params_at_construction)

  def stop_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    params_at_construction = query_function.keywords.get(
        'params_at_construction', {})
    params_at_construction['stop'] = query_record.stop
    return functools.partial(
        query_function,
        params_at_construction=params_at_construction)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'JSON response format is not supported for Grok. '
        'Code should never reach here.')

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'JSON schema response format is not supported for Grok. '
        'Code should never reach here.')

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return query_function

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    params_at_construction = query_function.keywords.get(
        'params_at_construction', {})
    params_at_construction['tools'] = [web_search()]
    return functools.partial(
        query_function,
        params_at_construction=params_at_construction)

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return response.content

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    raise Exception(
        'JSON response format is not supported for Grok. '
        'Code should never reach here.')

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    raise Exception(
        'JSON schema response format is not supported for Grok. '
        'Code should never reach here.')

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    return response[1]

  def _grok_chat_api_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> dict:

      params_at_construction = None
      params_after_construction = None

      if 'params_at_construction' in create.keywords:
        params_at_construction = create.keywords.get(
            'params_at_construction')
        del create.keywords['params_at_construction']

      if 'params_after_construction' in create.keywords:
        params_after_construction = create.keywords.get(
            'params_after_construction')
        del create.keywords['params_after_construction']

      # Responsible for temperature, stop, tools.
      if params_at_construction:
        create = functools.partial(create, **params_at_construction)
      create = create()

      # Responsible for messages, prompt, system.
      if params_after_construction:
        messages = params_after_construction.get('messages', [])
        if messages:
          for message in messages:
            create.append(message)
        del params_after_construction['messages']

      # Responsible for response format text, json, json schema, pydantic.
      if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        create = functools.partial(
            create.parse,
            query_record.response_format.value.class_value)
      else:
        create = functools.partial(create.sample)

      # Responsible for max_tokens.
      if params_after_construction:
        create = functools.partial(create, **params_after_construction)

      return create

  def generate_text_proc(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record.chosen_endpoint)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self.add_features_to_query_function(create, query_record)

    create = self._grok_chat_api_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

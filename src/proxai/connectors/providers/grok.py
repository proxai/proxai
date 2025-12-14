from typing import Any, Callable, Optional
import functools
import json
import os
from xai_sdk import Client
from xai_sdk.chat import user, system, assistant
from xai_sdk.tools import web_search
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class GrokConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'grok'

  def init_model(self):
    return Client(api_key=os.getenv("XAI_API_KEY"))

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

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
    return functools.partial(
        query_function,
        tools=[web_search()])

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
      chat_function: Callable,
      query_record: types.QueryRecord) -> dict:

    def _collect_params(**kwargs) -> dict:
      return kwargs
    params = self.add_features_to_query_function(
        functools.partial(_collect_params), query_record)()

    if params.get('temperature') is not None:
      chat_function = functools.partial(
          chat_function, temperature=params['temperature'])
    if params.get('stop') is not None:
      chat_function = functools.partial(
          chat_function, stop=params['stop'])
    if params.get('tools') is not None:
      chat_function = functools.partial(
          chat_function, tools=params['tools'])

    chat = chat_function()

    if params.get('system') is not None:
      chat.append(system(params['system']))
    if params.get('messages') is not None:
      for message in params['messages']:
        if message['role'] == 'system':
          chat.append(system(message['content']))
        if message['role'] == 'user':
          chat.append(user(message['content']))
        elif message['role'] == 'assistant':
          chat.append(assistant(message['content']))
    if params.get('prompt') is not None:
      chat.append(user(params['prompt']))

    if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      chat_sample = functools.partial(
          chat.parse,
          query_record.response_format.value.class_value)
    else:
      chat_sample = functools.partial(chat.sample)

    if params.get('max_tokens') is not None:
      chat_sample = functools.partial(
          chat_sample, max_tokens=params['max_tokens'])

    return chat_sample


  def generate_text_proc(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record.chosen_endpoint)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self._grok_chat_api_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

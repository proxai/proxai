import functools
from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.databricks_mock as databricks_mock
import proxai.types as types


class DatabricksConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'databricks'

  def init_model(self):
    w = WorkspaceClient()
    return w.serving_endpoints.get_open_ai_client()

  def init_mock_model(self):
    return databricks_mock.DatabricksMock()

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
    # Note: Databricks uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    # Some parameters may not work as expected for some models. For example,
    # the system instruction doesn't have any effect on the completion for
    # databricks-dbrx-instruct. But the stop parameter works as expected for
    # this model. However, system instruction works for
    # databricks-llama-2-70b-chat.
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

  def _add_json_guidance_to_user_message(
      self,
      query_function: Callable):
    # NOTE: Some Databricks models expect JSON guidance to be in the user message.
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
        'Web search is not supported for Databricks. Code should never reach here.')

  def _extract_text_from_content(self, content) -> str:
    # If content is already a string, return it directly
    if isinstance(content, str):
      return content

    # If content is a list of chunks, extract text from TextChunk objects
    if isinstance(content, list):
      text_parts = []
      for chunk in content:
        # Check for TextChunk (has type='text' and text attribute)
        chunk_type = getattr(chunk, 'type', None)
        if chunk_type == 'text':
          text = getattr(chunk, 'text', None)
          if text:
            text_parts.append(text)
      return '\n'.join(text_parts) if text_parts else ''

    # Fallback: try to convert to string
    return str(content) if content else ''

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return self._extract_text_from_content(response.choices[0].message.content)

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(
        self._extract_text_from_content(response.choices[0].message.content))

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(
        self._extract_text_from_content(response.choices[0].message.content))

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

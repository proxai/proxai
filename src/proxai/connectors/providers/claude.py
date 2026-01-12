import functools
from collections.abc import Callable
from typing import Any

import anthropic

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.claude_mock as claude_mock
import proxai.types as types

# Beta header required for structured outputs feature
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class ClaudeConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'claude'

  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return claude_mock.ClaudeMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'messages.create':
      return functools.partial(self.api.messages.create)
    elif chosen_endpoint == 'beta.messages.create':
      return functools.partial(
          self.api.beta.messages.create,
          betas=[STRUCTURED_OUTPUTS_BETA])
    elif chosen_endpoint == 'beta.messages.parse':
      return functools.partial(
          self.api.beta.messages.parse,
          betas=[STRUCTURED_OUTPUTS_BETA])
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
    return functools.partial(query_function, system=query_record.system)

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
    if isinstance(query_record.stop, str):
      return functools.partial(
          query_function,
          stop_sequences=[query_record.stop])
    else:
      return functools.partial(
          query_function,
          stop_sequences=query_record.stop)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
          'JSON response format is not supported for Claude. Code should '
          'never reach here.')

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    schema_value = query_record.response_format.value
    json_schema_obj = schema_value['json_schema']
    output_format = {
        'type': 'json_schema',
        'schema': json_schema_obj.get('schema', json_schema_obj)
    }
    return functools.partial(
        query_function,
        output_format=output_format)

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        output_format=query_record.response_format.value.class_value)

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(query_function, tools=[{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }])

  def _extract_text_from_content(self, content_blocks) -> str:
    # Extract text from content blocks
    # When web_search or other tools are used, response may contain multiple
    # block types (ServerToolUseBlock, TextBlock, etc). Find TextBlocks.
    text_parts = []
    for block in content_blocks:
      if hasattr(block, 'text'):
        text_parts.append(block.text)
    return '\n'.join(text_parts) if text_parts else ''

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return self._extract_text_from_content(response.content)

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(
        self._extract_text_from_content(response.content))

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(
        self._extract_text_from_content(response.content))

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    return response.parsed_output

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record.chosen_endpoint)

    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    create = self.add_features_to_query_function(create, query_record)

    # NOTE: Claude API requires max_tokens to be set.
    if create.keywords.get('max_tokens') is None:
      create = functools.partial(
          create,
          max_tokens=4096)

    response = create()

    return self.format_response_from_providers(response, query_record)

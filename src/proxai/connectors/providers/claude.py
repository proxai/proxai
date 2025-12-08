import anthropic
from typing import Any, Callable
import functools
import json
import re
import proxai.types as types
import proxai.connectors.providers.claude_mock as claude_mock
import proxai.connectors.model_connector as model_connector

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
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    # Choose the appropriate API method based on response format
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      # Use beta.messages.parse for Pydantic models
      return functools.partial(
          self.api.beta.messages.parse,
          betas=[STRUCTURED_OUTPUTS_BETA])
    elif (query_record.response_format is not None and
          query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      # Use beta.messages.create for JSON schema
      return functools.partial(
          self.api.beta.messages.create,
          betas=[STRUCTURED_OUTPUTS_BETA])
    else:
      # Use standard messages.create for text and simple JSON
      return functools.partial(self.api.messages.create)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: Claude uses 'user' and 'assistant' as roles. 'system' is a
    # different parameter.
    query_messages = []
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.system is not None:
      create = functools.partial(create, system=query_record.system)
    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    else:
      # Note: Claude models require a max_tokens parameter.
      create = functools.partial(create, max_tokens=4096)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        create = functools.partial(create, stop_sequences=[query_record.stop])
      else:
        create = functools.partial(create, stop_sequences=query_record.stop)
    if query_record.web_search is not None:
      create = functools.partial(create, tools=[{
          "type": "web_search_20250305",
          "name": "web_search",
          "max_uses": 5
      }])

    # Handle response format configuration
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        output_format = {
            'type': 'json_schema',
            'schema': json_schema_obj.get('schema', json_schema_obj)
        }
        create = functools.partial(create, output_format=output_format)
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        create = functools.partial(
            create,
            output_format=query_record.response_format.value.class_value)

    return create

  def _extract_text_from_content(self, content_blocks) -> str:
    # Extract text from content blocks
    # When web_search or other tools are used, response may contain multiple
    # block types (ServerToolUseBlock, TextBlock, etc). We need to find TextBlocks.
    text_parts = []
    for block in content_blocks:
      if hasattr(block, 'text'):
        text_parts.append(block.text)
    return '\n'.join(text_parts) if text_parts else ''

  def _response_mapping(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:
    if query_record.response_format is None:
      return types.Response(
          value=self._extract_text_from_content(response.content),
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=self._extract_text_from_content(response.content),
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=self._extract_json_from_text(
              self._extract_text_from_content(response.content)),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=self._extract_json_from_text(
              self._extract_text_from_content(response.content)),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=response.parsed_output),
          type=types.ResponseType.PYDANTIC)

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

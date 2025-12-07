import anthropic
import copy
import functools
import json
import re
import proxai.types as types
import proxai.connectors.providers.claude_mock as claude_mock
import proxai.connectors.model_connector as model_connector


# Beta header required for structured outputs feature
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


def _extract_json_from_text(text: str) -> dict:
  """Extract JSON from text that may contain markdown code blocks or other text.

  Tries multiple strategies:
  1. Direct JSON parse
  2. Extract from markdown code blocks (```json ... ``` or ``` ... ```)
  3. Find JSON object pattern in text
  """
  # Strategy 1: Try direct parse
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    pass

  # Strategy 2: Extract from markdown code blocks
  # Match ```json ... ``` or ``` ... ```
  code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
  matches = re.findall(code_block_pattern, text)
  for match in matches:
    try:
      return json.loads(match.strip())
    except json.JSONDecodeError:
      continue

  # Strategy 3: Find JSON object pattern
  # Look for content between first { and last }
  first_brace = text.find('{')
  last_brace = text.rfind('}')
  if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
    try:
      return json.loads(text[first_brace:last_brace + 1])
    except json.JSONDecodeError:
      pass

  # If all strategies fail, raise an error
  raise json.JSONDecodeError(
      f"Could not extract valid JSON from response",
      text,
      0)


def _extract_text_from_content(content_blocks) -> str:
  # Extract text from content blocks
  # When web_search or other tools are used, response may contain multiple
  # block types (ServerToolUseBlock, TextBlock, etc). We need to find TextBlocks.
  text_parts = []
  for block in content_blocks:
    if hasattr(block, 'text'):
      text_parts.append(block.text)
  return '\n'.join(text_parts) if text_parts else ''


class ClaudeConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'claude'

  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return claude_mock.ClaudeMock()

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    # Note: Claude uses 'user' and 'assistant' as roles. 'system' is a
    # different parameter.
    query_messages = []
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    provider_model = query_record.provider_model

    # Choose the appropriate API method based on response format
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      # Use beta.messages.parse for Pydantic models
      create = functools.partial(
          self.api.beta.messages.parse,
          model=provider_model.provider_model_identifier,
          messages=query_messages,
          betas=[STRUCTURED_OUTPUTS_BETA])
    elif (query_record.response_format is not None and
          query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      # Use beta.messages.create for JSON schema
      create = functools.partial(
          self.api.beta.messages.create,
          model=provider_model.provider_model_identifier,
          messages=query_messages,
          betas=[STRUCTURED_OUTPUTS_BETA])
    else:
      # Use standard messages.create for text and simple JSON
      create = functools.partial(
          self.api.messages.create,
          model=provider_model.provider_model_identifier,
          messages=query_messages)

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

    completion = create()

    if query_record.response_format is None:
      return types.Response(
          value=_extract_text_from_content(completion.content),
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=_extract_text_from_content(completion.content),
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=_extract_json_from_text(_extract_text_from_content(completion.content)),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=json.loads(_extract_text_from_content(completion.content)),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=completion.parsed_output),
          type=types.ResponseType.PYDANTIC)

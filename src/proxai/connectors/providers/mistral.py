import copy
import functools
import json
import os
import re
from mistralai import Mistral
from mistralai.models import ResponseFormat, JSONSchema
import proxai.types as types
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.connectors.model_connector as model_connector


def _extract_json_from_text(text: str) -> dict:
  """Extract JSON from text that may contain markdown code blocks or other text."""
  # Strategy 1: Try direct parse
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    pass

  # Strategy 2: Extract from markdown code blocks
  code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
  matches = re.findall(code_block_pattern, text)
  for match in matches:
    try:
      return json.loads(match.strip())
    except json.JSONDecodeError:
      continue

  # Strategy 3: Find JSON object pattern
  first_brace = text.find('{')
  last_brace = text.rfind('}')
  if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
    try:
      return json.loads(text[first_brace:last_brace + 1])
    except json.JSONDecodeError:
      pass

  raise json.JSONDecodeError(
      f"Could not extract valid JSON from response",
      text,
      0)


class MistralConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'mistral'

  def init_model(self):
    return Mistral(api_key=os.environ.get('MISTRAL_API_KEY'))

  def init_mock_model(self):
    return mistral_mock.MistralMock()

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system is not None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'user':
          query_messages.append({'role': 'user', 'content': message['content']})
        if message['role'] == 'assistant':
          query_messages.append({'role': 'assistant', 'content': message['content']})
    provider_model = query_record.provider_model

    # Determine if we need to use chat.parse for Pydantic
    use_parse = (
        query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC)

    if use_parse:
      # Use chat.parse for Pydantic models
      create = functools.partial(
          self.api.chat.parse,
          model=provider_model.provider_model_identifier,
          messages=query_messages,
          response_format=query_record.response_format.value.class_value)
    else:
      # Use chat.complete for other formats
      create = functools.partial(
          self.api.chat.complete,
          model=provider_model.provider_model_identifier,
          messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      create = functools.partial(create, stop=query_record.stop)

    # Handle response format configuration (for non-Pydantic formats)
    if not use_parse and query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        create = functools.partial(
            create,
            response_format=ResponseFormat(type='json_object'))
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        schema_name = json_schema_obj.get('name', 'response_schema')
        raw_schema = json_schema_obj.get('schema', json_schema_obj)
        json_schema = JSONSchema(
            name=schema_name,
            schema=raw_schema)
        create = functools.partial(
            create,
            response_format=ResponseFormat(type='json_schema', json_schema=json_schema))

    completion = create()

    if query_record.response_format is None:
      return types.Response(
          value=completion.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=completion.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=_extract_json_from_text(completion.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=json.loads(completion.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=completion.choices[0].message.parsed),
          type=types.ResponseType.PYDANTIC)

import copy
import functools
import json
import os
import re
from google import genai
from google.genai import types as genai_types
import proxai.types as types
import proxai.connectors.providers.gemini_mock as gemini_mock
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_configs as model_configs


def _clean_schema_for_gemini(schema: dict) -> dict:
  """Clean up JSON schema for Gemini API compatibility.

  Gemini's response_schema doesn't support certain JSON Schema features like
  'additionalProperties', 'strict', etc. This function recursively removes
  unsupported fields.
  """
  if not isinstance(schema, dict):
    return schema

  # Fields not supported by Gemini's Schema type
  unsupported_fields = {'additionalProperties', 'strict', '$schema', 'name'}

  cleaned = {}
  for key, value in schema.items():
    if key in unsupported_fields:
      continue
    elif key == 'properties' and isinstance(value, dict):
      # Recursively clean property schemas
      cleaned[key] = {
          prop_name: _clean_schema_for_gemini(prop_schema)
          for prop_name, prop_schema in value.items()
      }
    elif key == 'items' and isinstance(value, dict):
      # Clean array item schemas
      cleaned[key] = _clean_schema_for_gemini(value)
    elif isinstance(value, dict):
      cleaned[key] = _clean_schema_for_gemini(value)
    else:
      cleaned[key] = value

  return cleaned


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


class GeminiConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'gemini'

  def init_model(self):
    return genai.Client(
      api_key=os.environ['GEMINI_API_KEY']
    )

  def init_mock_model(self):
    return gemini_mock.GeminiMock()

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    # Note: Gemini uses 'user' and 'model' as roles.  'system_instruction' is a
    # different parameter.
    contents = []
    if query_record.prompt is not None:
      contents.append(genai_types.Content(
          parts=[genai_types.Part(text=query_record.prompt)], role='user'))
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'assistant':
          contents.append(genai_types.Content(
              parts=[genai_types.Part(text=message['content'])],
              role='model'))
        if message['role'] == 'user':
          contents.append(genai_types.Content(
              parts=[genai_types.Part(text=message['content'])],
              role='user'))

    config = genai_types.GenerateContentConfig()
    if query_record.system is not None:
      config.system_instruction = query_record.system
    if query_record.max_tokens is not None:
      config.max_output_tokens = query_record.max_tokens
    if query_record.temperature is not None:
      config.temperature = query_record.temperature
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        config.stop_sequences = [query_record.stop]
      else:
        config.stop_sequences = query_record.stop
    if query_record.web_search is not None:
      config.tools = [genai_types.Tool(
          google_search=genai_types.GoogleSearch())]

    # Handle response format configuration
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass  # Default text response
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        # Simple JSON mode - request JSON output
        config.response_mime_type = 'application/json'
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        # JSON Schema mode - use response_mime_type and response_schema
        config.response_mime_type = 'application/json'
        schema_value = query_record.response_format.value
        # Handle OpenAI-style json_schema format
        if 'json_schema' in schema_value:
          json_schema_obj = schema_value['json_schema']
          raw_schema = json_schema_obj.get('schema', json_schema_obj)
        else:
          raw_schema = schema_value
        # Clean schema to remove unsupported fields for Gemini
        config.response_schema = _clean_schema_for_gemini(raw_schema)
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        # Pydantic mode - use response_schema with Pydantic class
        config.response_mime_type = 'application/json'
        config.response_schema = query_record.response_format.value.class_value

    response = self.api.models.generate_content(
        model=self.provider_model.provider_model_identifier,
        config=config,
        contents=contents
    )

    # Handle response based on format type
    if query_record.response_format is None:
      return types.Response(
          value=response.text,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=response.text,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      # Parse JSON from response (may need extraction if model includes extra text)
      return types.Response(
          value=_extract_json_from_text(response.text),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      # Parse JSON from structured output response
      return types.Response(
          value=json.loads(response.text),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      # For Pydantic, parse JSON and validate with the Pydantic model
      pydantic_class = query_record.response_format.value.class_value
      instance = pydantic_class.model_validate_json(response.text)
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=instance),
          type=types.ResponseType.PYDANTIC)

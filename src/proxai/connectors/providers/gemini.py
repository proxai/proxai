import copy
from typing import Any, Callable
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


class GeminiConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'gemini'

  def init_model(self):
    return genai.Client(
      api_key=os.environ['GEMINI_API_KEY']
    )

  def init_mock_model(self):
    return gemini_mock.GeminiMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(self.api.models.generate_content)

  def _clean_schema_for_gemini(self, schema: dict) -> dict:
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
            prop_name: self._clean_schema_for_gemini(prop_schema)
            for prop_name, prop_schema in value.items()
        }
      elif key == 'items' and isinstance(value, dict):
        # Clean array item schemas
        cleaned[key] = self._clean_schema_for_gemini(value)
      elif isinstance(value, dict):
        cleaned[key] = self._clean_schema_for_gemini(value)
      else:
        cleaned[key] = value

    return cleaned

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

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

    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        config.response_mime_type = 'application/json'
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        config.response_mime_type = 'application/json'
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        raw_schema = json_schema_obj.get('schema', json_schema_obj)
        config.response_schema = self._clean_schema_for_gemini(raw_schema)
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        config.response_mime_type = 'application/json'
        config.response_schema = query_record.response_format.value.class_value

    return functools.partial(create, config=config, contents=contents)

  def _response_mapping(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:

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
      return types.Response(
          value=self._extract_json_from_text(response.text),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=self._extract_json_from_text(response.text),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      pydantic_class = query_record.response_format.value.class_value
      instance = pydantic_class.model_validate_json(response.text)
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=instance),
          type=types.ResponseType.PYDANTIC)

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

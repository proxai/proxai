import copy
from typing import Any, Callable, Optional
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
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'models.generate_content':
      return functools.partial(self.api.models.generate_content)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    contents = [
        genai_types.Content(
              parts=[genai_types.Part(text=query_record.prompt)], role='user')]
    return functools.partial(
        query_function,
        contents=contents)

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    contents = query_function.keywords.get('contents')
    if contents is None:
      contents = []

    for message in query_record.messages:
      if message['role'] == 'assistant':
        contents.append(genai_types.Content(
            parts=[genai_types.Part(text=message['content'])],
            role='model'))
      if message['role'] == 'user':
        contents.append(genai_types.Content(
            parts=[genai_types.Part(text=message['content'])],
            role='user'))
    return functools.partial(
        query_function,
        contents=contents)

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.system_instruction = query_record.system
    return functools.partial(query_function, config=config)

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.max_output_tokens = query_record.max_tokens
    return functools.partial(query_function, config=config)

  def temperature_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.temperature = query_record.temperature
    return functools.partial(query_function, config=config)

  def stop_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    if isinstance(query_record.stop, str):
      config.stop_sequences = [query_record.stop]
    else:
      config.stop_sequences = query_record.stop
    return functools.partial(query_function, config=config)

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

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    config = genai_types.GenerateContentConfig()
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.response_mime_type = 'application/json'
    return functools.partial(query_function, config=config)

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.response_mime_type = 'application/json'
    schema_value = query_record.response_format.value
    json_schema_obj = schema_value['json_schema']
    raw_schema = json_schema_obj.get('schema', json_schema_obj)
    config.response_schema = self._clean_schema_for_gemini(raw_schema)
    return functools.partial(query_function, config=config)

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.response_mime_type = 'application/json'
    config.response_schema = query_record.response_format.value.class_value
    return functools.partial(query_function, config=config)

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    config = query_function.keywords.get('config')
    if config is None:
      config = genai_types.GenerateContentConfig()
    config.tools = [genai_types.Tool(
          google_search=genai_types.GoogleSearch())]
    return functools.partial(query_function, config=config)

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return response.text

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.text)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.text)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    pydantic_class = query_record.response_format.value.class_value
    return pydantic_class.model_validate_json(response.text)

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

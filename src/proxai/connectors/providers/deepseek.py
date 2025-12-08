from typing import Any, Callable
import functools
import json
import os
from openai import OpenAI
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class DeepSeekConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'deepseek'

  def init_model(self):
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com")

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(self.api.chat.completions.create)

  def _strict_feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: DeepSeek uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    query_messages = []

    # Build system message, potentially with JSON schema guidance
    system_content = query_record.system
    schema_guidance = None

    # DeepSeek doesn't support json_schema type, so we add schema to prompt
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        raw_schema = json_schema_obj.get('schema', json_schema_obj)
        schema_guidance = (
            f"You must respond with valid JSON that follows this schema:\n"
            f"{json.dumps(raw_schema, indent=2)}")
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        pydantic_class = query_record.response_format.value.class_value
        schema = pydantic_class.model_json_schema()
        schema_guidance = (
            f"You must respond with valid JSON that follows this schema:\n"
            f"{json.dumps(schema, indent=2)}")

    if schema_guidance:
      if system_content:
        system_content = f"{system_content}\n\n{schema_guidance}"
      else:
        system_content = schema_guidance

    if system_content is not None:
      query_messages.append({'role': 'system', 'content': system_content})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(
          create, max_completion_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      create = functools.partial(create, stop=query_record.stop)

    # Handle response format configuration
    # Note: DeepSeek only supports 'json_object' type, not 'json_schema'
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type in (
          types.ResponseFormatType.JSON,
          types.ResponseFormatType.JSON_SCHEMA,
          types.ResponseFormatType.PYDANTIC):
        # All JSON modes use json_object (DeepSeek doesn't support json_schema)
        create = functools.partial(
            create,
            response_format={'type': 'json_object'})

    return create

  def _best_effort_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    return create

  def _response_mapping(
      self,
      response: Any,
      query_record: types.QueryRecord) -> types.Response:

    # Handle response based on format type
    if query_record.response_format is None:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=response.choices[0].message.content,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=json.loads(response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=json.loads(response.choices[0].message.content),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      # For Pydantic, parse JSON and validate with the Pydantic model
      pydantic_class = query_record.response_format.value.class_value
      instance = pydantic_class.model_validate_json(
          response.choices[0].message.content)
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=instance),
          type=types.ResponseType.PYDANTIC)


  def generate_text_proc(
      self,
      query_record: types.QueryRecord,
      best_effort_feature_mapping: bool = True) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._strict_feature_mapping(create, query_record)
    # create = self._best_effort_feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

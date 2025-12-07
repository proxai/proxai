from typing import Any, Callable
import functools
import json
import os
from openai import OpenAI
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class GrokConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'grok'

  def init_model(self):
    return OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1")

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    if (query_record.response_format is not None and
        query_record.response_format.type == types.ResponseFormatType.PYDANTIC):
      return functools.partial(self.api.beta.chat.completions.parse)
    else:
      return functools.partial(self.api.chat.completions.create)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: Grok uses OpenAI-compatible API with 'system', 'user', and
    # 'assistant' as roles.
    query_messages = []
    if query_record.system is not None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(
          create, max_completion_tokens=query_record.max_tokens)
    elif self.provider_model.model in [
        'grok-3-mini-beta', 'grok-3-mini-fast-beta']:
      # Note: There is a bug in the grok api that if max_completion_tokens is
      # not set, the response is empty string.
      # TODO: Remove this once the bug is fixed.
      create = functools.partial(create, max_completion_tokens=1000000)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        create = functools.partial(create, stop=[query_record.stop])
      else:
        create = functools.partial(create, stop=query_record.stop)

    # Handle response format configuration
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        create = functools.partial(
            create,
            response_format={'type': 'json_object'})
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        create = functools.partial(
            create,
            response_format=query_record.response_format.value)
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        create = functools.partial(
            create,
            response_format=query_record.response_format.value.class_value)

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
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=response.choices[0].message.parsed),
          type=types.ResponseType.PYDANTIC)

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

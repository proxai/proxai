from typing import Any, Callable
import functools
import json
import cohere
import proxai.types as types
import proxai.connectors.providers.cohere_api_mock as cohere_api_mock
import proxai.connectors.model_connector as model_connector


class CohereConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'cohere'

  def init_model(self):
    return cohere.Client()

  def init_mock_model(self):
    return cohere_api_mock.CohereMock()

  def _get_api_call_function(
      self,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(self.api.chat)

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    # Note: Cohere uses 'SYSTEM', 'USER', and 'CHATBOT' as roles. Additionally,
    # system instructions can be provided in two ways: preamble parameter and
    # chat_history 'SYSTEM' role. The difference is explained in the
    # documentation. The suggested way is to use the preamble parameter.
    query_messages = []
    prompt = query_record.prompt
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'user':
          query_messages.append(
              {'role': 'USER', 'message': message['content']})
        if message['role'] == 'assistant':
          query_messages.append(
              {'role': 'CHATBOT', 'message': message['content']})
      prompt = query_messages[-1]['message']
      del query_messages[-1]
    create = functools.partial(create, message=prompt)
    if query_messages:
      create = functools.partial(create, chat_history=query_messages)

    if query_record.system is not None:
      create = functools.partial(create, preamble=query_record.system)
    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, list):
        create = functools.partial(create, stop_sequences=query_record.stop)
      else:
        create = functools.partial(create, stop_sequences=[query_record.stop])

    # Handle response format configuration
    if query_record.response_format is not None:
      if query_record.response_format.type == types.ResponseFormatType.TEXT:
        pass
      elif query_record.response_format.type == types.ResponseFormatType.JSON:
        # Simple JSON mode
        create = functools.partial(
            create,
            response_format={'type': 'json_object'})
      elif query_record.response_format.type == types.ResponseFormatType.JSON_SCHEMA:
        # JSON Schema mode - Cohere uses 'json_object' type with 'schema' field
        schema_value = query_record.response_format.value
        json_schema_obj = schema_value['json_schema']
        raw_schema = json_schema_obj.get('schema', json_schema_obj)
        create = functools.partial(
            create,
            response_format={'type': 'json_object', 'schema': raw_schema})
      elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
        # Pydantic mode - extract schema from Pydantic class
        pydantic_class = query_record.response_format.value.class_value
        schema = pydantic_class.model_json_schema()
        create = functools.partial(
            create,
            response_format={'type': 'json_object', 'schema': schema})

    return create

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
          value=json.loads(response.text),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
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

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self._response_mapping(response, query_record)

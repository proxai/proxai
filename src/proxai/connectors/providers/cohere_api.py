from typing import Any, Callable, Optional
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

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    return functools.partial(query_function, preamble=system_message)

  def json_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    return functools.partial(
        query_function,
        response_format={'type': 'json_object'})

  def json_schema_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    schema_value = query_record.response_format.value
    json_schema_obj = schema_value['json_schema']
    raw_schema = json_schema_obj.get('schema', json_schema_obj)
    return functools.partial(
        query_function,
        response_format={'type': 'json_object', 'schema': raw_schema})

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    pydantic_class = query_record.response_format.value.class_value
    schema = pydantic_class.model_json_schema()
    return functools.partial(
        query_function,
        response_format={'type': 'json_object', 'schema': schema})

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

    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, list):
        create = functools.partial(create, stop_sequences=query_record.stop)
      else:
        create = functools.partial(create, stop_sequences=[query_record.stop])

    create = self.add_system_and_response_format_params(create, query_record)

    return create

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
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

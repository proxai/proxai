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
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat':
      return functools.partial(self.api.chat)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere uses 'message' parameter for the current prompt,
    # not 'messages' array like OpenAI-style APIs.
    return functools.partial(
        query_function,
        message=query_record.prompt)

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere uses 'SYSTEM', 'USER', and 'CHATBOT' as roles.
    # Uses 'chat_history' for previous messages and 'message' for current.
    # The last user message becomes the 'message' parameter.
    query_messages = []
    for message in query_record.messages:
      if message['role'] == 'user':
        query_messages.append(
            {'role': 'USER', 'message': message['content']})
      elif message['role'] == 'assistant':
        query_messages.append(
            {'role': 'CHATBOT', 'message': message['content']})

    # The last message becomes the current 'message' parameter
    current_message = query_messages[-1]['message']
    chat_history = query_messages[:-1]

    query_function = functools.partial(
        query_function,
        message=current_message)
    if chat_history:
      query_function = functools.partial(
          query_function,
          chat_history=chat_history)
    return query_function

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere system instructions can be provided in two ways:
    # preamble parameter and chat_history 'SYSTEM' role.
    # The suggested way is to use the preamble parameter.
    return functools.partial(query_function, preamble=query_record.system)

  def max_tokens_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        max_tokens=query_record.max_tokens)

  def temperature_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        temperature=query_record.temperature)

  def stop_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere uses 'stop_sequences' parameter (list), not 'stop'.
    if isinstance(query_record.stop, list):
      return functools.partial(
          query_function,
          stop_sequences=query_record.stop)
    else:
      return functools.partial(
          query_function,
          stop_sequences=[query_record.stop])

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
    # Note: Cohere doesn't have native pydantic support like OpenAI's parse.
    # We use json_object with schema and parse manually in format_pydantic_response.
    pydantic_class = query_record.response_format.value.class_value
    schema = pydantic_class.model_json_schema()
    return functools.partial(
        query_function,
        response_format={'type': 'json_object', 'schema': schema})

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'Web search is not supported for Cohere. Code should never reach here.')

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

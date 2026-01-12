import functools
from collections.abc import Callable
from typing import Any

import cohere

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.cohere_api_mock as cohere_api_mock
import proxai.types as types


class CohereConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'cohere'

  def init_model(self):
    return cohere.ClientV2()

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
    # Note: Cohere v2 API uses 'messages' array like OpenAI-style APIs.
    return functools.partial(
        query_function,
        messages=[{'role': 'user', 'content': query_record.prompt}])

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere v2 API uses 'user' and 'assistant' as roles (lowercase).
    # Uses a single 'messages' parameter like OpenAI-style APIs.
    query_messages = []
    for message in query_record.messages:
      if message['role'] == 'user':
        query_messages.append(
            {'role': 'user', 'content': message['content']})
      elif message['role'] == 'assistant':
        query_messages.append(
            {'role': 'assistant', 'content': message['content']})

    return functools.partial(
        query_function,
        messages=query_messages)

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    # Note: Cohere v2 API uses 'system' role in messages array.
    # We need to prepend the system message to existing messages.
    existing_messages = query_function.keywords.get('messages', [])
    system_message = {'role': 'system', 'content': query_record.system}
    updated_messages = [system_message] + existing_messages
    return functools.partial(query_function, messages=updated_messages)

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
    # We use json_object with schema and parse manually.
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

  def _extract_text_from_content(self, content_items: list) -> str:
    """Extract text from Cohere v2 API response content items.

    The content array can contain different types of items:
    - TextAssistantMessageResponseContentItem (type='text') - has .text
    - ThinkingAssistantMessageResponseContentItem (type='thinking') - no .text

    We need to find the item with type='text' to get the actual response.
    """
    for item in content_items:
      # Check if item has 'type' attribute and it's 'text'
      has_text_type = hasattr(item, 'type') and item.type == 'text'
      has_text_only = hasattr(item, 'text') and not hasattr(item, 'type')
      if has_text_type or has_text_only:
        return item.text

    # Fallback: try to find any item with a text attribute
    for item in content_items:
      if hasattr(item, 'text'):
        return item.text

    raise ValueError(
        f"Could not find text content in Cohere response. "
        f"Content types: {[type(item).__name__ for item in content_items]}")

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    # Cohere v2 API response structure: response.message.content is a list
    # with different content types (text, thinking, etc.)
    return self._extract_text_from_content(response.message.content)

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    text = self._extract_text_from_content(response.message.content)
    return self._extract_json_from_text(text)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    text = self._extract_text_from_content(response.message.content)
    return self._extract_json_from_text(text)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    text = self._extract_text_from_content(response.message.content)
    pydantic_class = query_record.response_format.value.class_value
    return pydantic_class.model_validate_json(text)

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

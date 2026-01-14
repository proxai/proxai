import functools
from collections.abc import Callable
from typing import Any

from huggingface_hub import InferenceClient

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types


class HuggingFaceConnector(model_connector.ProviderModelConnector):
  """Connector for Hugging Face Inference API models."""

  def get_provider_name(self):
    return 'huggingface'

  def get_required_provider_token_names(self) -> list[str]:
    return ['HF_TOKEN']

  def init_model(self):
    return InferenceClient(
        provider="auto",
        token=self.provider_token_value_map['HF_TOKEN'])

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'chat.completions.create':
      return functools.partial(self.api.chat.completions.create)
    else:
      raise Exception(f'Invalid endpoint: {chosen_endpoint}')

  def prompt_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(
        query_function,
        messages=[{'role': 'user', 'content': query_record.prompt}])

  def messages_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    messages = query_function.keywords.get('messages')
    if messages is None:
      return functools.partial(
          query_function,
          messages=query_record.messages)
    else:
      messages = query_record.messages + messages
      return functools.partial(
          query_function,
          messages=messages)

  def system_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord) -> Callable:
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.insert(0, {'role': 'system', 'content': query_record.system})
    return functools.partial(query_function, messages=messages)

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
    # Note: HuggingFace API expects stop to be a list.
    if isinstance(query_record.stop, str):
      return functools.partial(
          query_function,
          stop=[query_record.stop])
    else:
      return functools.partial(
          query_function,
          stop=query_record.stop)

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
    return functools.partial(
        query_function,
        response_format=query_record.response_format.value)

  def pydantic_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    # Note: HuggingFace doesn't have native pydantic support.
    # We use json_schema format and parse manually in format_pydantic_response.
    pydantic_class = query_record.response_format.value.class_value
    schema = pydantic_class.model_json_schema()
    return functools.partial(
        query_function,
        response_format={
            'type': 'json_schema',
            'json_schema': {
                'name': query_record.response_format.value.class_name,
                'schema': schema,
                'strict': True
            }
        })

  def web_search_feature_mapping(
      self,
      query_function: Callable,
      query_record: types.QueryRecord):
    raise Exception(
        'Web search is not supported for HuggingFace.')

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    return response.choices[0].message.content

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response.choices[0].message.content)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    pydantic_class = query_record.response_format.value.class_value
    return pydantic_class.model_validate_json(
        response.choices[0].message.content)

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

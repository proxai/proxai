from typing import Any, Callable
import functools
import json
import os
import requests
from typing import Any, Dict, List, Optional
import proxai.types as types
import proxai.connectors.providers.huggingface_mock as huggingface_mock
import proxai.connectors.model_connector as model_connector

_MODEL_URL_MAP = {
    'Qwen/Qwen3-32B': 'https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions',
    'deepseek-ai/DeepSeek-R1': 'https://router.huggingface.co/together/v1/chat/completions',
    'deepseek-ai/DeepSeek-V3': 'https://router.huggingface.co/together/v1/chat/completions',
    'google/gemma-2-2b-it': 'https://router.huggingface.co/nebius/v1/chat/completions',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': 'https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3.1-8B-Instruct/v1/chat/completions',
    'microsoft/phi-4': 'https://router.huggingface.co/nebius/v1/chat/completions'
}


class _HuggingFaceRequest:
  def __init__(self):
    self.headers = {
        'Authorization': f'Bearer {os.environ["HUGGINGFACE_API_KEY"]}'}

  def generate_content(
      self,
      messages: List[Dict[str, str]],
      model: str,
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None,
      stop: Optional[List[str]]=None,
      response_format: Optional[Dict[str, Any]]=None) -> str:
    payload = {
        'model': model,
        'messages': messages
    }
    if max_tokens is not None:
      payload['max_tokens'] = max_tokens
    if temperature is not None:
      payload['temperature'] = temperature
    if stop is not None:
      payload['stop'] = stop
    if response_format is not None:
      payload['response_format'] = response_format
    response = requests.post(
        _MODEL_URL_MAP[model],
        headers=self.headers,
        json=payload)
    if response.status_code != 200:
      raise Exception(
          f"HuggingFace API error {response.status_code}: {response.text}")
    response_text = response.json()['choices'][0]['message']['content']
    return response_text


class HuggingFaceConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'huggingface'

  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return huggingface_mock.HuggingFaceMock()

  def _get_api_call_function(
      self,
      chosen_endpoint: str) -> Callable:
    if chosen_endpoint == 'generate_content':
      return functools.partial(self.api.generate_content)
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
        'Web search is not supported for HuggingFace. Code should never reach here.')

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
    # Note: HuggingFace response is already extracted as a string
    # in _HuggingFaceRequest.generate_content
    return response

  def format_json_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response)

  def format_json_schema_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> dict:
    return self._extract_json_from_text(response)

  def format_pydantic_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> Any:
    pydantic_class = query_record.response_format.value.class_value
    # NOTE: Double JSON encode/decode to ensure the response is a valid JSON
    # object. This may slow down the response time but it's a workaround to
    # ensure the response is a valid JSON object not well supported
    # HuggingFace response format.
    return pydantic_class.model_validate_json(
        json.dumps(self._extract_json_from_text(response)))

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

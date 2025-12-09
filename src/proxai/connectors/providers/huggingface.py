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
      query_record: types.QueryRecord) -> Callable:
    return functools.partial(self.api.generate_content)

  def system_feature_mapping(
      self,
      query_function: Callable,
      system_message: Optional[str] = None) -> Callable:
    if system_message is None:
      return query_function
    messages = query_function.keywords.get('messages')
    if messages is None:
      raise Exception('Set messages parameter before adding system message.')
    messages.insert(0, {'role': 'system', 'content': system_message})
    return functools.partial(query_function, messages=messages)

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

  def _feature_mapping(
      self,
      create: Callable,
      query_record: types.QueryRecord) -> Callable:
    provider_model = query_record.provider_model
    create = functools.partial(
        create, model=provider_model.provider_model_identifier)

    query_messages = []
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    create = functools.partial(create, messages=query_messages)

    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        create = functools.partial(create, stop=[query_record.stop])
      else:
        create = functools.partial(create, stop=query_record.stop)

    create = self.add_system_and_response_format_params(create, query_record)

    return create

  def format_text_response_from_provider(
      self,
      response: Any,
      query_record: types.QueryRecord) -> str:
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
      self, query_record: types.QueryRecord) -> types.Response:
    create = self._get_api_call_function(query_record)

    create = self._feature_mapping(create, query_record)

    response = create()

    return self.format_response_from_providers(response, query_record)

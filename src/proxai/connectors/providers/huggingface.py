import copy
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

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
    provider_model = query_record.provider_model
    query_messages = []
    if query_record.system is not None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)

    create = functools.partial(
        self.api.generate_content,
        model=provider_model.provider_model_identifier,
        messages=query_messages)
    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
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
        # For Pydantic, build json_schema format from Pydantic class
        pydantic_class = query_record.response_format.value.class_value
        schema = pydantic_class.model_json_schema()
        create = functools.partial(
            create,
            response_format={
                'type': 'json_schema',
                'json_schema': {
                    'name': query_record.response_format.value.class_name,
                    'schema': schema,
                    'strict': True
                }
            })

    completion = create()

    # Handle response based on format type
    if query_record.response_format is None:
      return types.Response(
          value=completion,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.TEXT:
      return types.Response(
          value=completion,
          type=types.ResponseType.TEXT)
    elif query_record.response_format.type == types.ResponseFormatType.JSON:
      return types.Response(
          value=json.loads(completion),
          type=types.ResponseType.JSON)
    elif (query_record.response_format.type ==
          types.ResponseFormatType.JSON_SCHEMA):
      return types.Response(
          value=json.loads(completion),
          type=types.ResponseType.JSON)
    elif query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      # For Pydantic, parse JSON and validate with the Pydantic model
      pydantic_class = query_record.response_format.value.class_value
      instance = pydantic_class.model_validate_json(completion)
      return types.Response(
          value=types.ResponsePydanticValue(
              class_name=query_record.response_format.value.class_name,
              instance_value=instance),
          type=types.ResponseType.PYDANTIC)

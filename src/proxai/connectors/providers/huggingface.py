import copy
import functools
import os
import requests
from typing import Dict, List, Optional
import proxai.types as types
import proxai.connectors.providers.huggingface_mock as huggingface_mock
import proxai.connectors.model_connector as model_connector

_MODEL_URL_MAP = {
    'Qwen/Qwen2.5-Coder-32B-Instruct': 'https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions',
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
      stop: Optional[List[str]]=None) -> str:
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
    response = requests.post(
        _MODEL_URL_MAP[model],
        headers=self.headers,
        json=payload)
    response_text = response.json()['choices'][0]['message']['content']
    return response_text


class HuggingFaceConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'huggingface'

  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return huggingface_mock.HuggingFaceMock()

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return copy.deepcopy(query_record)

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
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
    completion = create()
    return completion

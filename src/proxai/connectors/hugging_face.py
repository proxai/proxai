import functools
import os
import requests
from typing import Union, Optional
import proxai.types as types
from .hugging_face_mock import HuggingFaceMock
from .model_connector import ModelConnector


class _HuggingFaceRequest:
  def __init__(self):
    self.api_url = 'https://api-inference.huggingface.co/models/'
    self.headers = {
        'Authorization': f'Bearer {os.environ["HUGGINGFACE_API_KEY"]}'}

  def generate_content(
      self,
      prompt: str,
      model: str,
      max_tokens: Optional[int]=None,
      temperature: Optional[int]=None) -> str:
    json_input = {'inputs': prompt}
    if max_tokens != None:
      json_input['max_new_tokens'] = max_tokens
    if temperature != None:
      json_input['temperature'] = temperature
    response = requests.post(
        self.api_url + model,
        headers=self.headers,
        json=json_input)
    text = response.json()[0]['generated_text']
    if text.startswith(prompt):
      text = text[len(prompt):]
    return text


class HuggingFaceConnector(ModelConnector):
  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return HuggingFaceMock()

  def generate_text_proc(
      self,
      model: types.ModelType,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = None,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None
  ) -> str:
    if system != None:
      raise ValueError('Hugging Face does not support system messages.')
    if messages != None:
      raise ValueError('Hugging Face does not support message history.')
    _, provider_model = model

    create = functools.partial(
        self.api.generate_content,
        model=provider_model)
    if max_tokens != None:
      # Note: Hugging Face uses max_new_tokens instead of max_tokens.
      # This implies that input tokens are not counted.
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      raise ValueError('Hugging Face does not support stop tokens.')

    completion = create(prompt)
    return completion

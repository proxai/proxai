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

  def generate_content(self, prompt: str, model: str) -> str:
    response = requests.post(
        self.api_url + model,
        headers=self.headers,
        json={'inputs': prompt})
    text = response.json()[0]['generated_text']
    if text.startswith(prompt):
      text = text[len(prompt):]
    return text


class HuggingFaceConnector(ModelConnector):
  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return HuggingFaceMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    response = self.api.generate_content(
        prompt, model=self.model_signature.model.value)
    return response

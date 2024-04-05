from typing import Union, Optional
import cohere
import proxai.types as types
from .cohere_api_mock import CohereMock
from .model_connector import ModelConnector


class CohereConnector(ModelConnector):
  def init_model(self):
    return cohere.Client()

  def init_mock_model(self):
    return CohereMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    response = self.api.chat(
        message=prompt,
        model=self.model_signature.model.value)
    return response.text

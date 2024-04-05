from typing import Union, Optional
import google.generativeai as genai
import proxai.types as types
from .gemini_mock import GeminiMock
from .model_connector import ModelConnector


class GeminiConnector(ModelConnector):
  def init_model(self):
    return genai.GenerativeModel(self.model_signature.model.value)

  def init_mock_model(self):
    return GeminiMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    response = self.api.generate_content(prompt)
    return response.text

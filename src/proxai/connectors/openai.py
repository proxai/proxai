from typing import Union, Optional
from openai import OpenAI
import proxai.types as types
from .openai_mock import OpenAIMock
from .model_connector import ModelConnector


class OpenAIConnector(ModelConnector):
  def init_model(self):
    return OpenAI()

  def init_mock_model(self):
    return OpenAIMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    completion = self.api.chat.completions.create(
        model=self.model_signature.model.value,
        messages=[
          {'role': 'system', 'content': 'You are an helpful assistant.'},
          {'role': 'user', 'content': prompt}
        ]
    )
    return completion.choices[0].message.content

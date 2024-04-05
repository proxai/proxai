from typing import Union, Optional
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import proxai.types as types
from .mistral_mock import MistralMock
from .model_connector import ModelConnector


class MistralConnector(ModelConnector):
  def init_model(self):
    return MistralClient()

  def init_mock_model(self):
    return MistralMock()

  def generate_text(self, prompt: str, max_tokens: int) -> str:
    response = self.api.chat(
        model=self.model_signature.model.value,
        messages=[
            ChatMessage(role='user', content=prompt)
        ],
    )
    return response.choices[0].message.content

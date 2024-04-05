import anthropic
from typing import Union, Optional
import proxai.types as types
from .claude_mock import ClaudeMock
from .model_connector import ModelConnector


class ClaudeConnector(ModelConnector):
  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return ClaudeMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    message = self.api.messages.create(
        model=self.model_signature.model.value,
        max_tokens=1024,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return message.content[0].text

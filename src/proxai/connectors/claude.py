import anthropic
import functools
from typing import Union, Optional
import proxai.types as types
from .claude_mock import ClaudeMock
from .model_connector import ModelConnector


class ClaudeConnector(ModelConnector):
  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return ClaudeMock()

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
    query_messages = []
    if prompt != None:
      query_messages.append({'role': 'user', 'content': prompt})
    if messages != None:
      query_messages.extend(messages)
    _, provider_model = model

    create = functools.partial(
        self.api.messages.create,
        model=provider_model,
        messages=query_messages)
    if system != None:
      create = functools.partial(create, system=system)
    if max_tokens != None:
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      create = functools.partial(create, stop_sequences=stop)

    completion = create()
    return completion.content[0].text

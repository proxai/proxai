import functools
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
    if system != None:
      query_messages.append({'role': 'system', 'content': system})
    if prompt != None:
      query_messages.append({'role': 'user', 'content': prompt})
    if messages != None:
      query_messages.extend(messages)
    _, provider_model = model

    create = functools.partial(
        self.api.chat.completions.create,
        model=provider_model,
        messages=query_messages)
    if max_tokens != None:
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      create = functools.partial(create, stop=stop)

    completion = create()
    return completion.choices[0].message.content

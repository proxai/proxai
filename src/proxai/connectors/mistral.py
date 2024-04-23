import functools
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
    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if system != None:
      query_messages.append(ChatMessage(role='system', content=system))
    if prompt != None:
      query_messages.append(ChatMessage(role='user', content=prompt))
    if messages != None:
      for message in messages:
        if message['role'] == 'user':
          query_messages.append(
              ChatMessage(role='user', content=message['content']))
        if message['role'] == 'assistant':
          query_messages.append(
              ChatMessage(role='assistant', content=message['content']))
    _, provider_model = model

    create = functools.partial(
        self.api.chat,
        model=provider_model,
        messages=query_messages)
    if max_tokens != None:
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      raise ValueError('Stop sequences are not supported by Mistral')

    completion = create()
    return completion.choices[0].message.content

  # def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
  #   response = self.api.chat(
  #       model=self.provider_model,
  #       messages=[
  #           ChatMessage(role='user', content=prompt)
  #       ],
  #   )
  #   return response.choices[0].message.content

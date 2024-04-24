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
      self, query_record: types.QueryRecord) -> str:
    # Note: Mistral uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system != None:
      query_messages.append(
          ChatMessage(role='system', content=query_record.system))
    if query_record.prompt != None:
      query_messages.append(
          ChatMessage(role='user', content=query_record.prompt))
    if query_record.messages != None:
      for message in query_record.messages:
        if message['role'] == 'user':
          query_messages.append(
              ChatMessage(role='user', content=message['content']))
        if message['role'] == 'assistant':
          query_messages.append(
              ChatMessage(role='assistant', content=message['content']))
    _, provider_model = query_record.model

    create = functools.partial(
        self.api.chat,
        model=provider_model,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      self.feature_fail(
          query_record=query_record,
          message='Stop sequences are not supported by Mistral')

    completion = create()
    return completion.choices[0].message.content

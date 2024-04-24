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
      self, query_record: types.QueryRecord) -> str:
    # Note: Claude uses 'user' and 'assistant' as roles. 'system' is a
    # different parameter.
    query_messages = []
    if query_record.prompt != None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages != None:
      query_messages.extend(query_record.messages)
    _, provider_model = query_record.model

    create = functools.partial(
        self.api.messages.create,
        model=provider_model,
        messages=query_messages)
    if query_record.system != None:
      create = functools.partial(create, system=query_record.system)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop_sequences=query_record.stop)

    completion = create()
    return completion.content[0].text

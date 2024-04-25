import copy
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

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return copy.deepcopy(query_record)

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: OpenAI uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system != None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt != None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages != None:
      query_messages.extend(query_record.messages)
    _, provider_model = query_record.model

    create = functools.partial(
        self.api.chat.completions.create,
        model=provider_model,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop=query_record.stop)

    completion = create()
    return completion.choices[0].message.content

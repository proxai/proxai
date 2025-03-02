import copy
import functools
import math
from typing import Union, Optional
from openai import OpenAI
import proxai.types as types
import proxai.connectors.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class OpenAIConnector(model_connector.ProviderModelConnector):
  def init_model(self):
    return OpenAI()

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return copy.deepcopy(query_record)

  def _get_token_count(self, logging_record: types.LoggingRecord):
    # Note: This temporary implementation is not accurate.
    # Better version should be calculated from the api response or at least
    # libraries like tiktoker.
    return logging_record.query_record.max_tokens

  def _get_query_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return 0

  def _get_response_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return logging_record.query_record.max_tokens

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: OpenAI uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system != None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt != None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages != None:
      query_messages.extend(query_record.messages)
    provider_model = query_record.provider_model

    create = functools.partial(
        self.api.chat.completions.create,
        model=provider_model.model,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop=query_record.stop)

    completion = create()
    return completion.choices[0].message.content

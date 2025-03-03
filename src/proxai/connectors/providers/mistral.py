import copy
import functools
import math
from typing import Union, Optional
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import proxai.types as types
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.connectors.model_connector as model_connector


class MistralConnector(model_connector.ProviderModelConnector):
  def init_model(self):
    return MistralClient()

  def init_mock_model(self):
    return mistral_mock.MistralMock()

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    query_record = copy.deepcopy(query_record)
    if query_record.stop != None:
      self.feature_fail(
          query_record=query_record,
          message='Stop sequences are not supported by Mistral')
      query_record.stop = None
    return query_record

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
    provider_model = query_record.provider_model

    create = functools.partial(
        self.api.chat,
        model=provider_model.model,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)

    completion = create()
    return completion.choices[0].message.content

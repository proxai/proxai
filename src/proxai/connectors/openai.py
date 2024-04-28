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

  def _get_estimated_price(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    # Needs to get updated all the time.
    # This is just a temporary implementation.
    query_token_count = self._get_query_token_count(logging_record)
    response_token_count = self._get_response_token_count(logging_record)
    _, provider_model = logging_record.query_record.model
    if provider_model == types.OpenAIModel.GPT_4:
      return ((query_token_count / 1000000) * 30.0
              + (response_token_count / 1000000) * 60.0)
    elif provider_model == types.OpenAIModel.GPT_4_TURBO_PREVIEW:
      return ((query_token_count / 1000000) * 10.0
              + (response_token_count / 1000000) * 30.0)
    elif provider_model == types.OpenAIModel.GPT_3_5_TURBO:
      return ((query_token_count / 1000000) * 0.5
              + (response_token_count / 1000000) * 1.5)
    elif provider_model == types.OpenAIModel.GPT_3_5_TURBO_INSTRUCT:
      return ((query_token_count / 1000000) * 1.5
              + (response_token_count / 1000000) * 2.0)
    elif provider_model == types.OpenAIModel.BABBAGE:
      return ((query_token_count / 1000000) * 0.4
              + (response_token_count / 1000000) * 0.4)
    elif provider_model == types.OpenAIModel.DAVINCI:
      return ((query_token_count / 1000000) * 2.0
              + (response_token_count / 1000000) * 2.0)
    else:
      raise ValueError(f'Model not found.\n{logging_record.query_record.model}')

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

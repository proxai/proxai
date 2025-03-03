import anthropic
import copy
import functools
import math
from typing import Union, Optional
import proxai.types as types
from .claude_mock import ClaudeMock
from .model_connector import ModelConnector


class ClaudeConnector(ModelConnector):
  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return ClaudeMock()

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

  def _get_estimated_cost(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    # Needs to get updated all the time.
    # This is just a temporary implementation.
    query_token_count = self._get_query_token_count(logging_record)
    response_token_count = self._get_response_token_count(logging_record)
    _, provider_model = logging_record.query_record.model
    if provider_model == types.ClaudeModel.CLAUDE_3_OPUS:
      return math.floor(query_token_count * 15.0 + response_token_count * 75.0)
    elif provider_model == types.ClaudeModel.CLAUDE_3_SONNET:
      return math.floor(query_token_count * 3.0 + response_token_count * 15.0)
    elif provider_model == types.ClaudeModel.CLAUDE_3_HAIKU:
      return math.floor(query_token_count * 0.25 + response_token_count * 1.25)
    else:
      raise ValueError(f'Model not found.\n{logging_record.query_record.model}')

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
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

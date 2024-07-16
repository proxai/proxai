import copy
import os
import functools
import math
from typing import Union, Optional, Type
from databricks_genai_inference import ChatCompletion
import proxai.types as types
from .databricks_mock import DatabricksMock
from .model_connector import ModelConnector


class DatabricksConnector(ModelConnector):
  def init_model(self):
    return ChatCompletion

  def init_mock_model(self):
    return DatabricksMock()

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
    if provider_model == types.DatabricksModel.LLAMA_3_70B_INSTRUCT:
      return math.floor(
          query_token_count * 14.28 + response_token_count * 42.85)
    elif provider_model == types.DatabricksModel.DBRX_INSTRUCT:
      return math.floor(
          query_token_count * 32.14 + response_token_count * 96.42)
    elif provider_model == types.DatabricksModel.MIXTRAL_8x7B_INSTRUCT:
      return math.floor(
          query_token_count * 21.42 + response_token_count * 21.42)
    elif provider_model in types.DatabricksModel.LLAMA_2_70B_CHAT:
      return math.floor(
          query_token_count * 28.57 + response_token_count * 28.57)
    elif provider_model in types.DatabricksModel.MPT_30B_INSTRUCT:
      return math.floor(
          query_token_count * 14.28 + response_token_count * 14.28)
    elif provider_model in types.DatabricksModel.MPT_7B_INSTRUCT:
      return math.floor(query_token_count * 7.14 + response_token_count * 7.14)
    elif provider_model in types.DatabricksModel.BGE_LARGE_EN:
      return math.floor(query_token_count * 1.42 + response_token_count * 1.42)
    else:
      raise ValueError(f'Model not found.\n{logging_record.query_record.model}')

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Databricks tries to use same parameters with OpenAI.
    # Some parameters seems not working as expected for some models. For
    # example, the system instruction doesn't have any effect on the completion
    # for databricks-dbrx-instruct. But the stop parameter works as expected for
    # this model. However, system instruction works for
    # databricks-llama-2-70b-chat.
    query_messages = []
    if query_record.system != None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt != None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages != None:
      query_messages.extend(query_record.messages)
    _, provider_model = query_record.model

    create = functools.partial(
        self.api.create,
        model=provider_model,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop=query_record.stop)

    completion = create()
    return completion.message

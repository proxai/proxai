import copy
import os
import functools
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

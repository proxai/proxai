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
    # Note: Databricks tries to use same parameters with OpenAI.
    # Some parameters seems not working as expected for some models. For
    # example, the system instruction doesn't have any effect on the completion
    # for databricks-dbrx-instruct. But the stop parameter works as expected for
    # this model. However, system instruction works for
    # databricks-llama-2-70b-chat.
    query_messages = []
    if system != None:
      query_messages.append({'role': 'system', 'content': system})
    if prompt != None:
      query_messages.append({'role': 'user', 'content': prompt})
    if messages != None:
      query_messages.extend(messages)
    _, provider_model = model

    create = functools.partial(
        self.api.create,
        model=provider_model,
        messages=query_messages)
    if max_tokens != None:
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      create = functools.partial(create, stop=stop)

    completion = create()
    return completion.message

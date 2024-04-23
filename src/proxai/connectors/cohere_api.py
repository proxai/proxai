import functools
from typing import Union, Optional
import cohere
import proxai.types as types
from .cohere_api_mock import CohereMock
from .model_connector import ModelConnector


class CohereConnector(ModelConnector):
  def init_model(self):
    return cohere.Client()

  def init_mock_model(self):
    return CohereMock()

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
    # Note: Cohere uses 'SYSTEM', 'USER', and 'CHATBOT' as roles. Additionally,
    # system instructions can be provided in two ways: preamble parameter and
    # chat_history 'SYSTEM' role. The difference is explained in the
    # documentation. The suggested way is to use the preamble parameter.
    query_messages = []
    if messages != None:
      for message in messages:
        if message['role'] == 'user':
          query_messages.append(
              {'role': 'USER', 'message': message['content']})
        if message['role'] == 'assistant':
          query_messages.append(
              {'role': 'CHATBOT', 'message': message['content']})
      prompt = query_messages[-1]['message']
      del query_messages[-1]
    _, provider_model = model

    create = functools.partial(
        self.api.chat,
        model=provider_model,
        message=prompt)
    if system != None:
      create = functools.partial(create, preamble=system)
    if query_messages:
      create = functools.partial(create, chat_history=query_messages)
    if max_tokens != None:
      create = functools.partial(create, max_tokens=max_tokens)
    if temperature != None:
      create = functools.partial(create, temperature=temperature)
    if stop != None:
      create = functools.partial(create, stop_sequences=stop)

    completion = create()
    return completion.text

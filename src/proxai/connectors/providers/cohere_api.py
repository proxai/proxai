import copy
import functools
import cohere
import proxai.types as types
import proxai.connectors.providers.cohere_api_mock as cohere_api_mock
import proxai.connectors.model_connector as model_connector


class CohereConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'cohere'

  def init_model(self):
    return cohere.Client()

  def init_mock_model(self):
    return cohere_api_mock.CohereMock()

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Cohere uses 'SYSTEM', 'USER', and 'CHATBOT' as roles. Additionally,
    # system instructions can be provided in two ways: preamble parameter and
    # chat_history 'SYSTEM' role. The difference is explained in the
    # documentation. The suggested way is to use the preamble parameter.
    query_messages = []
    prompt = query_record.prompt
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'user':
          query_messages.append(
              {'role': 'USER', 'message': message['content']})
        if message['role'] == 'assistant':
          query_messages.append(
              {'role': 'CHATBOT', 'message': message['content']})
      prompt = query_messages[-1]['message']
      del query_messages[-1]
    provider_model = query_record.provider_model

    create = functools.partial(
        self.api.chat,
        model=provider_model.provider_model_identifier,
        message=prompt)
    if query_record.system is not None:
      create = functools.partial(create, preamble=query_record.system)
    if query_messages:
      create = functools.partial(create, chat_history=query_messages)
    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, list):
        create = functools.partial(create, stop_sequences=query_record.stop)
      else:
        create = functools.partial(create, stop_sequences=[query_record.stop])

    completion = create()
    return completion.text

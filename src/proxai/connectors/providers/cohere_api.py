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

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return copy.deepcopy(query_record)

  def get_token_count(self, logging_record: types.LoggingRecord):
    # Note: This temporary implementation is not accurate.
    # Better version should be calculated from the api response or at least
    # libraries like tiktoker.
    return logging_record.query_record.max_tokens

  def get_query_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return 0

  def get_response_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return logging_record.query_record.max_tokens

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Cohere uses 'SYSTEM', 'USER', and 'CHATBOT' as roles. Additionally,
    # system instructions can be provided in two ways: preamble parameter and
    # chat_history 'SYSTEM' role. The difference is explained in the
    # documentation. The suggested way is to use the preamble parameter.
    query_messages = []
    prompt = query_record.prompt
    if query_record.messages != None:
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
        model=provider_model.model,
        message=prompt)
    if query_record.system != None:
      create = functools.partial(create, preamble=query_record.system)
    if query_messages:
      create = functools.partial(create, chat_history=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop != None:
      create = functools.partial(create, stop_sequences=query_record.stop)

    completion = create()
    return completion.text

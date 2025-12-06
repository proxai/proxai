import copy
import functools
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import proxai.types as types
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.connectors.model_connector as model_connector


class MistralConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'mistral'

  def init_model(self):
    return MistralClient()

  def init_mock_model(self):
    return mistral_mock.MistralMock()

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> types.Response:
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
        model=provider_model.provider_model_identifier,
        messages=query_messages)
    if query_record.max_tokens != None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)

    completion = create()
    return types.Response(
        value=completion.choices[0].message.content,
        type=types.ResponseType.TEXT)

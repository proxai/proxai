import copy
import functools
import os
from openai import OpenAI
import proxai.types as types
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.connectors.model_connector as model_connector


class GrokConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'grok'

  def init_model(self):
    return OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1")

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

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
    # Note: OpenAI uses 'system', 'user', and 'assistant' as roles.
    query_messages = []
    if query_record.system is not None:
      query_messages.append({'role': 'system', 'content': query_record.system})
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    provider_model = query_record.provider_model

    create = functools.partial(
        self.api.chat.completions.create,
        model=provider_model.provider_model_identifier,
        messages=query_messages)
    if query_record.max_tokens is not None:
      create = functools.partial(
          create, max_completion_tokens=query_record.max_tokens)
    elif self.provider_model.model in [
        'grok-3-mini-beta', 'grok-3-mini-fast-beta']:
      # Note: There is a bug in the grok api that if max_completion_tokens is
      # not set, the response is empty string.
      create = functools.partial(create, max_completion_tokens=1000000)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        create = functools.partial(create, stop=[query_record.stop])
      else:
        create = functools.partial(create, stop=query_record.stop)

    completion = create()
    return completion.choices[0].message.content

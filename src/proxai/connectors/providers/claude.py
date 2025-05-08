import anthropic
import copy
import functools
import proxai.types as types
import proxai.connectors.providers.claude_mock as claude_mock
import proxai.connectors.model_connector as model_connector


class ClaudeConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'claude'

  def init_model(self):
    return anthropic.Anthropic()

  def init_mock_model(self):
    return claude_mock.ClaudeMock()

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Claude uses 'user' and 'assistant' as roles. 'system' is a
    # different parameter.
    query_messages = []
    if query_record.prompt is not None:
      query_messages.append({'role': 'user', 'content': query_record.prompt})
    if query_record.messages is not None:
      query_messages.extend(query_record.messages)
    provider_model = query_record.provider_model

    create = functools.partial(
        self.api.messages.create,
        model=provider_model.provider_model_identifier,
        messages=query_messages)
    if query_record.system is not None:
      create = functools.partial(create, system=query_record.system)
    if query_record.max_tokens is not None:
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    else:
      # Note: Claude models require a max_tokens parameter.
      create = functools.partial(create, max_tokens=4096)
    if query_record.temperature is not None:
      create = functools.partial(create, temperature=query_record.temperature)
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        create = functools.partial(create, stop_sequences=[query_record.stop])
      else:
        create = functools.partial(create, stop_sequences=query_record.stop)

    completion = create()
    return completion.content[0].text

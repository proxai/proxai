import functools

from xai_sdk import Client
from xai_sdk.chat import assistant, system, user
from xai_sdk.proto import chat_pb2

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.grok_mock as grok_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class GrokConnector(model_connector.ProviderModelConnector):
  """Connector for xAI Grok models."""

  def init_model(self):
    return Client(api_key=self.provider_token_value_map['XAI_API_KEY'])

  def init_mock_model(self):
    return grok_mock.GrokMock()

  PROVIDER_NAME = 'grok'

  PROVIDER_API_KEYS = ['XAI_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat.create',
  ]

  # The xAI SDK only exposes `'low'` and `'high'` reasoning effort levels, so
  # MEDIUM is mapped to `'high'` instead of being dropped silently.
  _REASONING_EFFORT = {
      types.ThinkingType.LOW: 'low',
      types.ThinkingType.MEDIUM: 'high',
      types.ThinkingType.HIGH: 'high',
  }

  ENDPOINT_CONFIG = {
      'chat.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  def _build_create_kwargs(
      self, query_record: types.QueryRecord) -> dict:
    """Build the keyword arguments for `client.chat.create`."""
    kwargs = {
        'model': query_record.provider_model.provider_model_identifier,
    }

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        kwargs['max_tokens'] = query_record.parameters.max_tokens

      if query_record.parameters.temperature is not None:
        kwargs['temperature'] = query_record.parameters.temperature

      if query_record.parameters.stop is not None:
        if isinstance(query_record.parameters.stop, str):
          kwargs['stop'] = [query_record.parameters.stop]
        else:
          kwargs['stop'] = query_record.parameters.stop

      if query_record.parameters.thinking is not None:
        kwargs['reasoning_effort'] = self._REASONING_EFFORT[
            query_record.parameters.thinking]

    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        kwargs['tools'] = [chat_pb2.Tool(web_search=chat_pb2.WebSearch())]
        kwargs['include'] = ['web_search_call_output', 'inline_citations']

    if query_record.response_format.type == types.ResponseFormatType.JSON:
      kwargs['response_format'] = 'json_object'

    return kwargs

  def _build_messages(
      self, query_record: types.QueryRecord) -> list:
    """Translate the query record into xAI SDK message protos."""
    messages = []

    if query_record.system_prompt is not None:
      messages.append(system(query_record.system_prompt))

    if query_record.chat is not None:
      if 'system_prompt' in query_record.chat:
        messages.append(system(query_record.chat['system_prompt']))
      for message in query_record.chat['messages']:
        role = message['role']
        content = message['content']
        if isinstance(content, list):
          # The proxai chat schema lets `content` be a list of typed parts
          # (text/image/...). Grok's helpers only accept plain strings, so
          # collapse text parts and skip everything else.
          text_parts = [
              p['text'] for p in content
              if isinstance(p, dict) and p.get('type') == 'text']
          content = '\n'.join(text_parts)
        if role == 'user':
          messages.append(user(content))
        elif role == 'assistant':
          messages.append(assistant(content))
        elif role == 'system':
          messages.append(system(content))

    if query_record.prompt is not None:
      messages.append(user(query_record.prompt))

    return messages

  def _run_chat(
      self,
      kwargs: dict,
      messages: list,
      pydantic_class):
    """Open a chat, append messages, and run sample/parse."""
    chat = self.api.chat.create(**kwargs)
    for message in messages:
      chat.append(message)
    if pydantic_class is not None:
      response, parsed = chat.parse(pydantic_class)
      return (response, parsed)
    return (chat.sample(), None)

  def _chat_create_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    kwargs = self._build_create_kwargs(query_record)
    messages = self._build_messages(query_record)

    pydantic_class = None
    if (query_record.response_format is not None
        and query_record.response_format.type
        == types.ResponseFormatType.PYDANTIC):
      pydantic_class = query_record.response_format.pydantic_class

    response, result_record = self._safe_provider_query(
        functools.partial(
            self._run_chat, kwargs, messages, pydantic_class))
    if result_record.error is not None:
      return result_record

    sample_response, parsed = response

    parsed_content = []

    if (sample_response.reasoning_content is not None
        and len(sample_response.reasoning_content) > 0):
      parsed_content.append(
          message_content.MessageContent(
              type=message_content.ContentType.THINKING,
              text=sample_response.reasoning_content,
          )
      )

    if pydantic_class is not None:
      parsed_content.append(
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name=pydantic_class.__name__,
                  class_value=pydantic_class,
                  instance_value=parsed,
              ),
          )
      )
    else:
      parsed_content.append(
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text=sample_response.content,
          )
      )

    if (query_record.tools is not None
        and types.Tools.WEB_SEARCH in query_record.tools):
      citations = []
      for citation_url in (sample_response.citations or []):
        citations.append(
            message_content.Citation(
                title=None,
                url=citation_url,
            )
        )
      parsed_content.append(
          message_content.MessageContent(
              type=message_content.ContentType.TOOL,
              tool_content=message_content.ToolContent(
                  name='web_search',
                  kind=message_content.ToolKind.RESULT,
                  citations=citations,
              ),
          )
      )

    result_record.content = parsed_content
    return result_record

  ENDPOINT_EXECUTORS = {
      'chat.create': '_chat_create_executor',
  }

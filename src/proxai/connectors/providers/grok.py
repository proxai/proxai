import base64
import functools

from xai_sdk import Client
from xai_sdk.chat import assistant, image, system, text, user
from xai_sdk.proto import chat_pb2

import proxai.connectors.content_utils as content_utils
import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.grok_mock as grok_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class GrokConnector(provider_connector.ProviderConnector):
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
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.BEST_EFFORT,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
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

    if query_record.output_format.type == types.OutputFormatType.JSON:
      kwargs['response_format'] = 'json_object'

    return kwargs

  @staticmethod
  def _to_grok_content(part_dict):
    """Convert a ProxAI content block to an xAI SDK Content proto.

    Type mapping:
      text     → ``text()`` proto
      image    → ``image()`` proto with URL or data URI.
                 Supports PNG and JPEG only.
      document → Text-based docs (md, csv, txt) are read and sent
                 as ``text()`` protos via ``content_utils``. Binary
                 formats (PDF, docx, xlsx) are dropped because the
                 SDK only supports file references by pre-uploaded
                 file_id, not inline data.

    Returns None for unsupported content types.
    """
    content_type = part_dict.get('type')
    if content_type == 'text':
      return text(part_dict['text'])
    if content_type == 'image':
      if 'source' in part_dict:
        return image(part_dict['source'])
      mime_type = part_dict.get('media_type', 'image/png')
      if 'data' in part_dict:
        return image(f"data:{mime_type};base64,{part_dict['data']}")
      if 'path' in part_dict:
        with open(part_dict['path'], 'rb') as f:
          encoded = base64.b64encode(f.read()).decode('utf-8')
        return image(f"data:{mime_type};base64,{encoded}")
      return None
    if content_type == 'document':
      text_content = content_utils.read_text_document(part_dict)
      if text_content is not None:
        return text(text_content)
      pdf_content = content_utils.read_pdf_document(part_dict)
      if pdf_content is not None:
        return text(pdf_content)
      return None
    return None

  def _build_messages(
      self, query_record: types.QueryRecord) -> list:
    """Translate the query record into xAI SDK message protos."""
    messages = []

    if query_record.system_prompt is not None:
      messages.append(system(query_record.system_prompt))

    if query_record.chat is not None:
      if 'system_prompt' in query_record.chat:
        messages.append(system(query_record.chat['system_prompt']))
      for msg in query_record.chat['messages']:
        role = msg['role']
        content = msg['content']
        if isinstance(content, str):
          parts = [text(content)]
        elif isinstance(content, list):
          parts = []
          for block in content:
            part = self._to_grok_content(block)
            if part is not None:
              parts.append(part)
        else:
          parts = [text(str(content))]
        if role == 'user':
          messages.append(user(*parts))
        elif role == 'assistant':
          messages.append(assistant(*parts))
        elif role == 'system':
          messages.append(system(*parts))

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
      query_record: types.QueryRecord) -> types.ExecutorResult:
    kwargs = self._build_create_kwargs(query_record)
    messages = self._build_messages(query_record)

    pydantic_class = None
    if (query_record.output_format is not None
        and query_record.output_format.type
        == types.OutputFormatType.PYDANTIC):
      pydantic_class = query_record.output_format.pydantic_class

    response, result_record = self._safe_provider_query(
        functools.partial(
            self._run_chat, kwargs, messages, pydantic_class))
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

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
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
      'chat.create': '_chat_create_executor',
  }

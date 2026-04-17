import functools

from mistralai import Mistral
from mistralai.models.websearchtool import WebSearchTool

import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class MistralConnector(provider_connector.ProviderConnector):
  """Connector for Mistral AI models."""

  def init_model(self):
    return Mistral(api_key=self.provider_token_value_map['MISTRAL_API_KEY'])

  def init_mock_model(self):
    return mistral_mock.MistralMock()

  PROVIDER_NAME = 'mistral'

  PROVIDER_API_KEYS = ['MISTRAL_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat.complete',
      'chat.parse',
      'beta.conversations.start',
  ]

  ENDPOINT_CONFIG = {
      'chat.complete': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.SUPPORTED,
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
      'chat.parse': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.SUPPORTED,
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
      'beta.conversations.start': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.BEST_EFFORT,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
  }

  def _add_common_params(self, call, query_record: types.QueryRecord):
    """Add messages, system, and parameter kwargs to a Mistral chat call."""
    if query_record.prompt is not None:
      call = functools.partial(
          call, messages=[{'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      call = functools.partial(call, messages=query_record.chat['messages'])

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        call = functools.partial(
            call, max_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        call = functools.partial(
            call, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        call = functools.partial(call, stop=query_record.parameters.stop)

      if query_record.parameters.n is not None:
        call = functools.partial(call, n=query_record.parameters.n)

    return call

  def _parse_message_content(self, content) -> list:
    """Parse a Mistral message.content value into MessageContent blocks.

    Magistral reasoning models return a list of ThinkChunk/TextChunk objects;
    standard models return a plain string. Both shapes land here.
    """
    if isinstance(content, str):
      return [
          message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text=content,
          )
      ]

    parsed = []
    if isinstance(content, list):
      for chunk in content:
        chunk_type = getattr(chunk, 'type', None)
        if chunk_type == 'thinking':
          inner = getattr(chunk, 'thinking', None) or []
          text_parts = []
          for inner_chunk in inner:
            inner_text = getattr(inner_chunk, 'text', None)
            if inner_text:
              text_parts.append(inner_text)
          parsed.append(
              message_content.MessageContent(
                  type=message_content.ContentType.THINKING,
                  text='\n'.join(text_parts),
              )
          )
        elif chunk_type == 'text':
          text = getattr(chunk, 'text', None) or ''
          parsed.append(
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT,
                  text=text,
              )
          )
    return parsed

  def _chat_complete_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    create = functools.partial(self.api.chat.complete)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    create = self._add_common_params(create, query_record)

    if query_record.output_format.type == types.OutputFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    result_record.content = self._parse_message_content(
        response.choices[0].message.content)

    if response.choices is not None and len(response.choices) > 1:
      result_record.choices = []
      for choice in response.choices:
        result_record.choices.append(
            types.ChoiceType(
                content=self._parse_message_content(choice.message.content)
            )
        )
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _chat_parse_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    create = functools.partial(self.api.chat.parse)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    create = self._add_common_params(create, query_record)

    create = functools.partial(
        create, response_format=query_record.output_format.pydantic_class)

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.PYDANTIC_INSTANCE,
            pydantic_content=message_content.PydanticContent(
                class_name=query_record.output_format.pydantic_class.__name__,
                class_value=query_record.output_format.pydantic_class,
                instance_value=response.choices[0].message.parsed,
            ),
        )
    ]

    if response.choices is not None and len(response.choices) > 1:
      result_record.choices = []
      for choice in response.choices:
        result_record.choices.append(
            types.ChoiceType(
                content=[
                    message_content.MessageContent(
                        type=message_content.ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=message_content.PydanticContent(
                            class_name=query_record.output_format.pydantic_class.__name__,
                            class_value=query_record.output_format.pydantic_class,
                            instance_value=choice.message.parsed,
                        ),
                    )
                ]
            )
        )
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _beta_conversations_start_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    """Execute a Mistral beta.conversations.start call with web search.

    This endpoint is structurally different from chat.complete/chat.parse:
      - `inputs` replaces `messages` and takes a plain string prompt
      - `instructions` is the dedicated system-prompt kwarg
      - sampling params live inside a `completion_args` dict
      - tools accept typed SDK objects (WebSearchTool, etc.)
      - the response is a ConversationResponse with an `outputs` list that
        interleaves ToolExecutionEntry (breadcrumb of a tool call) and
        MessageOutputEntry (assistant message with inline TextChunk and
        ToolReferenceChunk for citations)
    The executor is intentionally self-contained; it does not share helpers
    with _chat_complete_executor.
    """
    start = functools.partial(self.api.beta.conversations.start)
    start = functools.partial(start, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      start = functools.partial(start, inputs=query_record.prompt)

    if query_record.system_prompt is not None:
      start = functools.partial(
          start, instructions=query_record.system_prompt)

    tool_objects = []
    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        tool_objects.append(WebSearchTool(type='web_search'))
    if tool_objects:
      start = functools.partial(start, tools=tool_objects)

    completion_args = {}
    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        completion_args['max_tokens'] = query_record.parameters.max_tokens
      if query_record.parameters.temperature is not None:
        completion_args['temperature'] = query_record.parameters.temperature
      if query_record.parameters.stop is not None:
        completion_args['stop'] = query_record.parameters.stop
    if completion_args:
      start = functools.partial(start, completion_args=completion_args)

    response, result_record = self._safe_provider_query(start)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    parsed = []
    for output in response.outputs or []:
      output_type = getattr(output, 'type', None)

      if output_type == 'message.output':
        content = getattr(output, 'content', None)
        # When the model doesn't invoke a tool, `content` is a plain string
        # instead of a list of chunks. Both shapes land here.
        if isinstance(content, str):
          parsed.append(
              message_content.MessageContent(
                  type=message_content.ContentType.TEXT,
                  text=content,
              )
          )
          continue

        citations = []
        for chunk in content or []:
          chunk_type = getattr(chunk, 'type', None)
          if chunk_type == 'text':
            parsed.append(
                message_content.MessageContent(
                    type=message_content.ContentType.TEXT,
                    text=getattr(chunk, 'text', '') or '',
                )
            )
          elif chunk_type == 'tool_reference':
            citations.append(
                message_content.Citation(
                    title=getattr(chunk, 'title', None),
                    url=getattr(chunk, 'url', None),
                )
            )
        if citations:
          parsed.append(
              message_content.MessageContent(
                  type=message_content.ContentType.TOOL,
                  tool_content=message_content.ToolContent(
                      name='web_search',
                      kind=message_content.ToolKind.RESULT,
                      citations=citations,
                  ),
              )
          )

    # Mistral's beta.conversations.start has no native JSON / pydantic mode, so
    # json and pydantic are declared BEST_EFFORT. The framework injects schema
    # guidance into the prompt, but the model wraps its output in markdown
    # fences — `result_adapter`'s raw `json.loads` would fail on that. Pre-clean
    # TEXT blocks through the base-class helper so result_adapter receives a
    # JSON block it can pass through (for JSON) or validate (for PYDANTIC).
    needs_json_extract = (
        query_record.output_format is not None
        and query_record.output_format.type in (
            types.OutputFormatType.JSON,
            types.OutputFormatType.PYDANTIC,
        )
    )
    if needs_json_extract:
      parsed = [
          message_content.MessageContent(
              type=message_content.ContentType.JSON,
              json=self._extract_json_from_text(c.text),
          ) if c.type == message_content.ContentType.TEXT else c
          for c in parsed
      ]

    result_record.content = parsed
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
    'chat.complete': '_chat_complete_executor',
    'chat.parse': '_chat_parse_executor',
    'beta.conversations.start': '_beta_conversations_start_executor',
  }

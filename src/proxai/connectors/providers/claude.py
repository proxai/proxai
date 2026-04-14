import functools

import anthropic

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.claude_mock as claude_mock
import proxai.types as types
import proxai.chat.message_content as message_content

# Beta header required for structured outputs feature
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class ClaudeConnector(model_connector.ProviderModelConnector):
  """Connector for Anthropic Claude models."""

  def init_model(self):
    return anthropic.Anthropic(
        api_key=self.provider_token_value_map['ANTHROPIC_API_KEY']
    )

  def init_mock_model(self):
    return claude_mock.ClaudeMock()

  PROVIDER_NAME = 'claude'

  PROVIDER_API_KEYS = ['ANTHROPIC_API_KEY']

  ENDPOINT_PRIORITY = [
      'beta.messages.stream',
  ]

  _THINKING_BUDGETS = {
      types.ThinkingType.LOW: 1024,
      types.ThinkingType.MEDIUM: 8192,
      types.ThinkingType.HIGH: 24576,
  }
  _DEFAULT_MAX_TOKENS = 32768

  ENDPOINT_CONFIG = {
      'beta.messages.stream': FeatureConfigType(
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

  def _add_common_params(
      self, partial_call, query_record: types.QueryRecord):
    """Add prompt, messages, system, and parameters to the streaming call."""
    if query_record.prompt is not None:
      partial_call = functools.partial(
          partial_call, messages=[
              {'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      partial_call = functools.partial(
          partial_call, messages=query_record.chat['messages'])
      if 'system_prompt' in query_record.chat:
        partial_call = functools.partial(
            partial_call, system=query_record.chat['system_prompt'])

    if query_record.system_prompt is not None:
      partial_call = functools.partial(
          partial_call, system=query_record.system_prompt)

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        partial_call = functools.partial(
            partial_call, max_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        partial_call = functools.partial(
            partial_call, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        if isinstance(query_record.parameters.stop, str):
          partial_call = functools.partial(
              partial_call, stop_sequences=[query_record.parameters.stop])
        else:
          partial_call = functools.partial(
              partial_call, stop_sequences=query_record.parameters.stop)

    thinking_budget = None
    if (query_record.parameters is not None
        and query_record.parameters.thinking is not None):
      thinking_budget = self._THINKING_BUDGETS[
          query_record.parameters.thinking]
      partial_call = functools.partial(
          partial_call, thinking={
              'type': 'enabled', 'budget_tokens': thinking_budget})

    # Claude API requires max_tokens to be set, and when thinking is enabled
    # it must be strictly greater than thinking.budget_tokens.
    if 'max_tokens' in partial_call.keywords:
      if (thinking_budget is not None
          and partial_call.keywords['max_tokens'] <= thinking_budget):
        raise ValueError(
            'Claude requires max_tokens > thinking budget_tokens. '
            f'Got max_tokens={partial_call.keywords["max_tokens"]}, '
            f'thinking budget={thinking_budget} '
            f'(ThinkingType.{query_record.parameters.thinking.value}). '
            'Increase max_tokens or lower the thinking level.')
    else:
      partial_call = functools.partial(
          partial_call, max_tokens=self._DEFAULT_MAX_TOKENS)

    return partial_call

  def _run_stream(self, stream_partial):
    """Open a streaming SDK call and return the accumulated final message.

    Anthropic requires streaming for any request that may take longer than
    ~10 min (large `max_tokens`, extended thinking). We always stream and
    accumulate so the rest of the connector can work with the same final
    Message shape as the non-streaming path.
    """
    with stream_partial() as stream:
      return stream.get_final_message()

  def _parse_content_blocks(self, response) -> list:
    """Parse Claude response content blocks into MessageContent list."""
    parsed = []
    for block in response.content:
      if hasattr(block, 'type') and block.type == 'thinking':
        parsed.append(
            message_content.MessageContent(
                type=message_content.ContentType.THINKING,
                text=block.thinking,
            )
        )
      elif hasattr(block, 'type') and block.type == 'web_search_tool_result':
        citations = []
        if isinstance(block.content, list):
          for result in block.content:
            citations.append(
                message_content.Citation(
                    title=result.title,
                    url=result.url,
                ))
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
      elif hasattr(block, 'text'):
        if getattr(block, 'citations', None):
          citations = []
          for citation in block.citations:
            if getattr(citation, 'type', None) == 'web_search_result_location':
              citations.append(
                  message_content.Citation(
                      title=citation.title,
                      url=citation.url,
                  ))
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
        parsed.append(
            message_content.MessageContent(
                type=message_content.ContentType.TEXT,
                text=block.text,
            )
        )
    return parsed

  def _beta_messages_stream_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    stream = functools.partial(self.api.beta.messages.stream)
    stream = functools.partial(stream, model=(
        query_record.provider_model.provider_model_identifier
    ))

    stream = self._add_common_params(stream, query_record)

    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        stream = functools.partial(
            stream, tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }])

    needs_structured_output = (
        query_record.response_format is not None
        and query_record.response_format.type
        == types.ResponseFormatType.PYDANTIC)

    if needs_structured_output:
      stream = functools.partial(
          stream,
          betas=[STRUCTURED_OUTPUTS_BETA],
          output_format=query_record.response_format.pydantic_class)

    response, result_record = self._safe_provider_query(
        functools.partial(self._run_stream, stream))
    if result_record.error is not None:
      return result_record

    if needs_structured_output:
      result_record.content = [
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name=query_record.response_format.pydantic_class.__name__,
                  class_value=query_record.response_format.pydantic_class,
                  instance_value=response.parsed_output,
              ),
          )
      ]
      for block in response.content:
        if hasattr(block, 'type') and block.type == 'thinking':
          result_record.content.insert(0,
              message_content.MessageContent(
                  type=message_content.ContentType.THINKING,
                  text=block.thinking,
              )
          )
    else:
      result_record.content = self._parse_content_blocks(response)

    return result_record

  ENDPOINT_EXECUTORS = {
    'beta.messages.stream': '_beta_messages_stream_executor',
  }

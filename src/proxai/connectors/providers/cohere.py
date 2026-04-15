import functools

import cohere

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.cohere_mock as cohere_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class CohereConnector(model_connector.ProviderModelConnector):
  """Connector for Cohere V2 chat models."""

  def init_model(self):
    return cohere.ClientV2(
        api_key=self.provider_token_value_map['CO_API_KEY'])

  def init_mock_model(self):
    return cohere_mock.CohereMock()

  PROVIDER_NAME = 'cohere'

  PROVIDER_API_KEYS = ['CO_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat',
  ]

  # Cohere's reasoning surface accepts a token budget under
  # `thinking={'type': 'enabled', 'token_budget': N}`. We mirror Claude's
  # ladder so users get comparable depth across providers.
  _THINKING_BUDGETS = {
      types.ThinkingType.LOW: 1024,
      types.ThinkingType.MEDIUM: 8192,
      types.ThinkingType.HIGH: 24576,
  }

  ENDPOINT_CONFIG = {
      'chat': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          # Cohere V2 has no separate system field — it expects a
          # {'role': 'system', ...} entry at the head of `messages`.
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              # Cohere V2 chat exposes no `n` parameter.
              n=FeatureSupportType.NOT_SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              # V2 chat has no built-in web search tool. Function tools
              # exist but are out of scope for the proxai web_search flag.
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              # `response_format={'type': 'json_object'}` is a native flag.
              json=FeatureSupportType.SUPPORTED,
              # Pydantic is upgraded to SUPPORTED because Cohere enforces a
              # JSON schema server-side via `response_format.json_schema`.
              # The SDK still returns the JSON as text (no `parsed` field),
              # so the executor runs `model_validate_json` itself before
              # emitting a PYDANTIC_INSTANCE block.
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  def _chat_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    create = functools.partial(self.api.chat)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[{'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      create = functools.partial(
          create, messages=query_record.chat['messages'])

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        if isinstance(query_record.parameters.stop, str):
          create = functools.partial(
              create, stop_sequences=[query_record.parameters.stop])
        else:
          create = functools.partial(
              create, stop_sequences=query_record.parameters.stop)

      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create, thinking={
                'type': 'enabled',
                'token_budget': self._THINKING_BUDGETS[
                    query_record.parameters.thinking],
            })

    needs_pydantic = (
        query_record.response_format is not None
        and query_record.response_format.type
        == types.ResponseFormatType.PYDANTIC)

    if query_record.response_format.type == types.ResponseFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    if needs_pydantic:
      create = functools.partial(
          create,
          response_format={
              'type': 'json_object',
              'json_schema': query_record.response_format.pydantic_class
                  .model_json_schema(),
          })

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record

    parsed = []
    text_buffer = []
    for item in response.message.content or []:
      item_type = getattr(item, 'type', None)
      if item_type == 'thinking':
        parsed.append(
            message_content.MessageContent(
                type=message_content.ContentType.THINKING,
                text=getattr(item, 'thinking', '') or '',
            )
        )
      elif item_type == 'text':
        text = getattr(item, 'text', '') or ''
        text_buffer.append(text)
        parsed.append(
            message_content.MessageContent(
                type=message_content.ContentType.TEXT,
                text=text,
            )
        )

    if needs_pydantic:
      pydantic_class = query_record.response_format.pydantic_class
      joined_text = ''.join(text_buffer)
      instance = pydantic_class.model_validate_json(joined_text)
      # Replace TEXT blocks with a single PYDANTIC_INSTANCE block, keeping
      # any THINKING blocks ahead of it so users still see reasoning.
      parsed = [
          c for c in parsed
          if c.type != message_content.ContentType.TEXT
      ]
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.PYDANTIC_INSTANCE,
              pydantic_content=message_content.PydanticContent(
                  class_name=pydantic_class.__name__,
                  class_value=pydantic_class,
                  instance_value=instance,
              ),
          )
      )

    result_record.content = parsed
    return result_record

  ENDPOINT_EXECUTORS = {
      'chat': '_chat_executor',
  }

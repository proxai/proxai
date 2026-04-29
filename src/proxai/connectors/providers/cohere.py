from __future__ import annotations

import base64
import functools

import cohere

import proxai.connectors.content_utils as content_utils
import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.cohere_mock as cohere_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class CohereConnector(provider_connector.ProviderConnector):
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
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.BEST_EFFORT,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
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

  @staticmethod
  def _build_data_uri(part_dict):
    """Build a ``data:<mime>;base64,...`` URI from a content block."""
    mime_type = part_dict.get('media_type', 'application/octet-stream')
    if 'data' in part_dict:
      return f"data:{mime_type};base64,{part_dict['data']}"
    if 'path' in part_dict:
      with open(part_dict['path'], 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
      return f"data:{mime_type};base64,{encoded}"
    return None

  @staticmethod
  def _to_cohere_part(part_dict):
    """Convert a ProxAI content block to a Cohere V2 chat content part.

    Type mapping:
      text     → ``{"type": "text", "text": "..."}``
      image    → ``{"type": "image_url", "image_url": {"url": "..."}}``
                 Supports PNG, JPEG, WebP, and non-animated GIF.
      document → Text-based docs (md, csv, txt) are read as text.
                 PDF is extracted via pypdf. Other binary formats
                 (docx, xlsx) are dropped. Cohere has no native
                 document input support.

    Returns None for unsupported content types.
    """
    content_type = part_dict.get('type')
    # Text
    if content_type == 'text':
      return {'type': 'text', 'text': part_dict['text']}
    # Image: URL or inline data URI
    if content_type == 'image':
      if 'source' in part_dict:
        url = part_dict['source']
      else:
        url = CohereConnector._build_data_uri(part_dict)
      if url is None:
        return None
      return {'type': 'image_url', 'image_url': {'url': url}}
    # Document: text extraction only (no native document support)
    if content_type == 'document':
      text_content = content_utils.read_text_document(part_dict)
      if text_content is not None:
        return {'type': 'text', 'text': text_content}
      pdf_content = content_utils.read_pdf_document(part_dict)
      if pdf_content is not None:
        return {'type': 'text', 'text': pdf_content}
      return None
    return None

  @staticmethod
  def _convert_messages(messages):
    """Convert ProxAI message content blocks to Cohere V2 format.

    String content is passed through unchanged. List content is
    converted block-by-block; blocks where the converter returns
    None are dropped.
    """
    converted = []
    for message in messages:
      if isinstance(message['content'], str):
        converted.append(message)
        continue
      if isinstance(message['content'], list):
        parts = []
        for block in message['content']:
          part = CohereConnector._to_cohere_part(block)
          if part is not None:
            parts.append(part)
        converted.append({**message, 'content': parts})
      else:
        converted.append(message)
    return converted

  def _chat_executor(
      self,
      query_record: types.QueryRecord) -> types.ExecutorResult:
    create = functools.partial(self.api.chat)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[{'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      messages = self._convert_messages(query_record.chat['messages'])
      create = functools.partial(create, messages=messages)

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
        query_record.output_format is not None
        and query_record.output_format.type
        == types.OutputFormatType.PYDANTIC)

    if query_record.output_format.type == types.OutputFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    if needs_pydantic:
      create = functools.partial(
          create,
          response_format={
              'type': 'json_object',
              'json_schema': query_record.output_format.pydantic_class
                  .model_json_schema(),
          })

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

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
      pydantic_class = query_record.output_format.pydantic_class
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
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
      'chat': '_chat_executor',
  }

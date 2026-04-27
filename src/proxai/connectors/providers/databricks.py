from __future__ import annotations

import base64
import functools

from databricks.sdk import WorkspaceClient

import proxai.chat.message_content as message_content
import proxai.connectors.content_utils as content_utils
import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.databricks_mock as databricks_mock
import proxai.types as types

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class DatabricksConnector(provider_connector.ProviderConnector):
  """Connector for Databricks model serving endpoints.

  Databricks exposes its foundation model serving endpoints through an
  OpenAI-compatible API surface. We obtain a pre-configured `openai.OpenAI`
  client via `WorkspaceClient.serving_endpoints.get_open_ai_client()` and
  drive two endpoints:

  - `chat.completions.create`: the default path. Handles plain text, JSON,
    thinking, and pydantic-as-prompt-injection. Reasoning models
    (`databricks-gpt-oss-*`) return `message.content` as a list of
    `{'type': 'reasoning', ...}` / `{'type': 'text', ...}` blocks; the
    executor parses both shapes.
  - `beta.chat.completions.parse`: server-side pydantic structured
    outputs for non-reasoning models. The OpenAI SDK's client-side parse
    step assumes `message.content` is a string, so reasoning models must
    NOT route here — they crash with a Pydantic ValidationError before
    `message.parsed` is populated. Keep reasoning-model pydantic routed
    through `chat.completions.create` by capping their model-level
    `response_format.pydantic = BEST_EFFORT` in `model_configs_data`.
  """

  def init_model(self):
    workspace = WorkspaceClient(
        host=self.provider_token_value_map['DATABRICKS_HOST'],
        token=self.provider_token_value_map['DATABRICKS_TOKEN'],
    )
    return workspace.serving_endpoints.get_open_ai_client()

  def init_mock_model(self):
    return databricks_mock.DatabricksMock()

  PROVIDER_NAME = 'databricks'

  PROVIDER_API_KEYS = ['DATABRICKS_TOKEN', 'DATABRICKS_HOST']

  ENDPOINT_PRIORITY = [
      'chat.completions.create',
      'beta.chat.completions.parse',
  ]

  ENDPOINT_CONFIG = {
      'chat.completions.create':
          FeatureConfigType(
              prompt=FeatureSupportType.SUPPORTED,
              messages=FeatureSupportType.SUPPORTED,
              system_prompt=FeatureSupportType.SUPPORTED,
              add_system_to_messages=True,
              parameters=ParameterConfigType(
                  max_tokens=FeatureSupportType.SUPPORTED,
                  temperature=FeatureSupportType.SUPPORTED,
                  stop=FeatureSupportType.SUPPORTED,
                  n=FeatureSupportType.SUPPORTED,
                  thinking=FeatureSupportType.SUPPORTED,
              ),
              tools=ToolConfigType(
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
                  json=FeatureSupportType.SUPPORTED,
                  pydantic=FeatureSupportType.BEST_EFFORT,
              ),
          ),
      'beta.chat.completions.parse':
          FeatureConfigType(
              prompt=FeatureSupportType.SUPPORTED,
              messages=FeatureSupportType.SUPPORTED,
              system_prompt=FeatureSupportType.SUPPORTED,
              add_system_to_messages=True,
              parameters=ParameterConfigType(
                  max_tokens=FeatureSupportType.SUPPORTED,
                  temperature=FeatureSupportType.SUPPORTED,
                  stop=FeatureSupportType.SUPPORTED,
                  n=FeatureSupportType.SUPPORTED,
                  thinking=FeatureSupportType.SUPPORTED,
              ),
              tools=ToolConfigType(
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
                  text=FeatureSupportType.NOT_SUPPORTED,
                  json=FeatureSupportType.NOT_SUPPORTED,
                  pydantic=FeatureSupportType.SUPPORTED,
              ),
          ),
  }

  def _parse_message_content(self, content) -> list:
    """Parse a Databricks chat message content into MessageContent blocks.

    Non-reasoning models return a plain string. Reasoning models
    (databricks-gpt-oss-*) return a list of dicts with `type` fields —
    `'reasoning'` (with a `summary` list of `{'type': 'summary_text',
    'text': ...}` items) and `'text'` (with a `text` field).
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
        chunk_type = None
        if isinstance(chunk, dict):
          chunk_type = chunk.get('type')
          if chunk_type == 'reasoning':
            summary_parts = []
            for summary in chunk.get('summary') or []:
              text = None
              if isinstance(summary, dict):
                text = summary.get('text')
              else:
                text = getattr(summary, 'text', None)
              if text:
                summary_parts.append(text)
            parsed.append(
                message_content.MessageContent(
                    type=message_content.ContentType.THINKING,
                    text='\n'.join(summary_parts),
                )
            )
          elif chunk_type == 'text':
            parsed.append(
                message_content.MessageContent(
                    type=message_content.ContentType.TEXT,
                    text=chunk.get('text') or '',
                )
            )
        else:
          chunk_type = getattr(chunk, 'type', None)
          if chunk_type == 'reasoning':
            summary_parts = []
            for summary in getattr(chunk, 'summary', None) or []:
              text = getattr(summary, 'text', None)
              if text:
                summary_parts.append(text)
            parsed.append(
                message_content.MessageContent(
                    type=message_content.ContentType.THINKING,
                    text='\n'.join(summary_parts),
                )
            )
          elif chunk_type == 'text':
            parsed.append(
                message_content.MessageContent(
                    type=message_content.ContentType.TEXT,
                    text=getattr(chunk, 'text', '') or '',
                )
            )
    return parsed

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
  def _to_databricks_part(part_dict):
    """Convert a ProxAI content block to a Databricks content part.

    Uses OpenAI-compatible format (same as OpenAI chat.completions).

    Type mapping:
      text     → ``{"type": "text", "text": "..."}``
      image    → ``{"type": "image_url", "image_url": {"url": "..."}}``
                 Supports JPEG, PNG, WebP, GIF.
      document → Text-based docs (md, csv, txt) are read as text.
                 PDF is extracted via pypdf. Other binary formats
                 (docx, xlsx) are dropped. Databricks has no native
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
        url = DatabricksConnector._build_data_uri(part_dict)
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
    """Convert ProxAI message content blocks to Databricks format.

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
          part = DatabricksConnector._to_databricks_part(block)
          if part is not None:
            parts.append(part)
        converted.append({**message, 'content': parts})
      else:
        converted.append(message)
    return converted

  def _chat_completions_create_executor(
      self, query_record: types.QueryRecord
  ) -> types.ExecutorResult:
    create = functools.partial(self.api.chat.completions.create)
    create = functools.partial(
        create, model=(query_record.provider_model.provider_model_identifier)
    )

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[{
              'role': 'user',
              'content': query_record.prompt
          }]
      )

    if query_record.chat is not None:
      messages = self._convert_messages(query_record.chat['messages'])
      create = functools.partial(create, messages=messages)

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_tokens=query_record.parameters.max_tokens
        )

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature
        )

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)

      if query_record.parameters.n is not None:
        create = functools.partial(create, n=query_record.parameters.n)

      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create,
            reasoning_effort=query_record.parameters.thinking.value.lower()
        )

    # Databricks' OpenAI-compatible chat.completions.create exposes the
    # native `{'type': 'json_object'}` response format. Pydantic stays at
    # BEST_EFFORT but we flip on json_object when a schema was requested so
    # the framework's downstream json.loads + model_validate step is
    # reliable.
    if query_record.output_format.type in (
        types.OutputFormatType.JSON,
        types.OutputFormatType.PYDANTIC,
    ):
      create = functools.partial(
          create, response_format={'type': 'json_object'}
      )

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    result_record.content = self._parse_message_content(
        response.choices[0].message.content
    )

    if response.choices is not None and len(response.choices) > 1:
      result_record.choices = []
      for choice in response.choices[1:]:
        result_record.choices.append(
            types.ChoiceType(
                content=self._parse_message_content(choice.message.content)
            )
        )

    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _beta_chat_completions_parse_executor(
      self, query_record: types.QueryRecord
  ) -> types.ExecutorResult:
    create = functools.partial(self.api.beta.chat.completions.parse)
    create = functools.partial(
        create, model=(query_record.provider_model.provider_model_identifier)
    )

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[{
              'role': 'user',
              'content': query_record.prompt
          }]
      )

    if query_record.chat is not None:
      messages = self._convert_messages(query_record.chat['messages'])
      create = functools.partial(create, messages=messages)

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_tokens=query_record.parameters.max_tokens
        )

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature
        )

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)

      if query_record.parameters.n is not None:
        create = functools.partial(create, n=query_record.parameters.n)

      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create,
            reasoning_effort=query_record.parameters.thinking.value.lower()
        )

    create = functools.partial(
        create, response_format=query_record.output_format.pydantic_class
    )

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    pydantic_class = query_record.output_format.pydantic_class
    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.PYDANTIC_INSTANCE,
            pydantic_content=message_content.PydanticContent(
                class_name=pydantic_class.__name__,
                class_value=pydantic_class,
                instance_value=response.choices[0].message.parsed,
            ),
        )
    ]

    if response.choices is not None and len(response.choices) > 1:
      result_record.choices = []
      for choice in response.choices[1:]:
        result_record.choices.append(
            types.ChoiceType(
                content=[
                    message_content.MessageContent(
                        type=message_content.ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=message_content.PydanticContent(
                            class_name=pydantic_class.__name__,
                            class_value=pydantic_class,
                            instance_value=choice.message.parsed,
                        ),
                    )
                ]
            )
        )

    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
      'chat.completions.create': '_chat_completions_create_executor',
      'beta.chat.completions.parse': '_beta_chat_completions_parse_executor',
  }

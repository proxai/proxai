from __future__ import annotations

import base64
import functools
import os

from openai import OpenAI

import proxai.connectors.content_utils as content_utils
import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.deepseek_mock as deepseek_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class DeepSeekConnector(provider_connector.ProviderConnector):
  """Connector for DeepSeek models.

  DeepSeek exposes an OpenAI-compatible HTTP API, so we reuse the official
  `openai` SDK with a base_url override. Only the `chat.completions.create`
  endpoint is implemented — DeepSeek does not currently expose a native
  structured-outputs parse endpoint, web search tool, or `n > 1` sampling.
  """

  def init_model(self):
    return OpenAI(
        api_key=self.provider_token_value_map['DEEPSEEK_API_KEY'],
        base_url='https://api.deepseek.com',
    )

  def init_mock_model(self):
    return deepseek_mock.DeepSeekMock()

  PROVIDER_NAME = 'deepseek'

  PROVIDER_API_KEYS = ['DEEPSEEK_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat.completions.create',
  ]

  ENDPOINT_CONFIG = {
      'chat.completions.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          add_system_to_messages=True,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.SUPPORTED,
              # The DeepSeek API rejects `n > 1` with a 400 error, so we
              # surface this as NOT_SUPPORTED rather than silently dropping it.
              n=FeatureSupportType.NOT_SUPPORTED,
              # DeepSeek has no `reasoning_effort` knob. The deepseek-reasoner
              # model always emits reasoning_content; deepseek-chat never does.
              # BEST_EFFORT lets the framework drop the parameter silently.
              thinking=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
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
  }

  @staticmethod
  def _to_deepseek_part(part_dict):
    """Convert a ProxAI content block to a DeepSeek content part.

    Type mapping:
      text     → ``{"type": "text", "text": "..."}``
      document → Text-based docs (md, csv, txt) are read as text.
                 PDF is extracted via pypdf. Other binary formats
                 (docx, xlsx) are dropped. DeepSeek has no native
                 document or image support.

    Returns None for unsupported content types.
    """
    content_type = part_dict.get('type')
    # Text
    if content_type == 'text':
      return {'type': 'text', 'text': part_dict['text']}
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
    """Convert ProxAI message content blocks to DeepSeek format.

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
          part = DeepSeekConnector._to_deepseek_part(block)
          if part is not None:
            parts.append(part)
        converted.append({**message, 'content': parts})
      else:
        converted.append(message)
    return converted

  def _chat_completions_create_executor(
      self,
      query_record: types.QueryRecord) -> types.ExecutorResult:
    create = functools.partial(self.api.chat.completions.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[
              {'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      messages = self._convert_messages(query_record.chat['messages'])
      create = functools.partial(create, messages=messages)

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_completion_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)

    # DeepSeek's chat.completions.create supports a native JSON object mode.
    # Pydantic stays at BEST_EFFORT (no native parse endpoint), but we can
    # still flip on json_object so the framework's downstream
    # json.loads + model_validate is reliable.
    if query_record.output_format.type in (
        types.OutputFormatType.JSON,
        types.OutputFormatType.PYDANTIC,
    ):
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    parsed = []
    message = response.choices[0].message

    # deepseek-reasoner exposes its chain of thought via `reasoning_content`.
    # deepseek-chat leaves the attribute unset / None.
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
      parsed.append(
          message_content.MessageContent(
              type=message_content.ContentType.THINKING,
              text=reasoning,
          )
      )

    parsed.append(
        message_content.MessageContent(
            type=message_content.ContentType.TEXT,
            text=message.content,
        )
    )

    result_record.content = parsed
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
      'chat.completions.create': '_chat_completions_create_executor',
  }

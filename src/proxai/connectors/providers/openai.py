import base64
import functools
import os
import time
import datetime

from openai import OpenAI

import proxai.connectors.content_utils as content_utils
import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
OutputFormatConfigType = types.OutputFormatConfigType


class OpenAIConnector(provider_connector.ProviderConnector):
  """Connector for OpenAI models."""

  def init_model(self):
    return OpenAI(api_key=self.provider_token_value_map['OPENAI_API_KEY'])

  def init_mock_model(self):
    return openai_mock.OpenAIMock()

  PROVIDER_NAME = 'openai'

  PROVIDER_API_KEYS = ['OPENAI_API_KEY']

  ENDPOINT_PRIORITY = [
      'chat.completions.create',
      'beta.chat.completions.parse',
      'responses.create',
      'images.generate',
      'audio.speech.create',
      'videos.create',
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
              n=FeatureSupportType.SUPPORTED,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.SUPPORTED,
              audio=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
      'beta.chat.completions.parse': FeatureConfigType(
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
              document=FeatureSupportType.SUPPORTED,
              audio=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.NOT_SUPPORTED,
              json=FeatureSupportType.NOT_SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
      'responses.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.SUPPORTED,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.BEST_EFFORT,
              thinking=FeatureSupportType.SUPPORTED,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              image=FeatureSupportType.SUPPORTED,
              document=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.BEST_EFFORT,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
          output_format=OutputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
      'images.generate': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              image=FeatureSupportType.SUPPORTED,
          ),
      ),
      'audio.speech.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              audio=FeatureSupportType.SUPPORTED,
          ),
      ),
      'videos.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          output_format=OutputFormatConfigType(
              video=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  _MIME_TO_AUDIO_FORMAT = {
      'audio/mpeg': 'mp3',
      'audio/wav': 'wav',
      'audio/flac': 'flac',
      'audio/aac': 'aac',
      'audio/ogg': 'ogg',
  }

  @staticmethod
  def _build_data_uri(part_dict):
    """Build a ``data:<mime>;base64,...`` URI from a content block.

    Reads from the ``data`` field (already base64-encoded) or the
    ``path`` field (reads and encodes the file). Returns None if
    neither field is present.
    """
    mime_type = part_dict.get('media_type', 'application/octet-stream')
    if 'data' in part_dict:
      return f"data:{mime_type};base64,{part_dict['data']}"
    if 'path' in part_dict:
      with open(part_dict['path'], 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
      return f"data:{mime_type};base64,{encoded}"
    return None

  @staticmethod
  def _to_chat_completions_part(part_dict):
    """Convert a ProxAI content block to a chat.completions content part.

    Used by ``chat.completions.create`` and
    ``beta.chat.completions.parse`` endpoints.

    Type mapping:
      text     → ``{"type": "text", "text": "..."}``
      image    → ``{"type": "image_url", "image_url": {"url": "..."}}``
      audio    → ``{"type": "input_audio", "input_audio":
                 {"data": "...", "format": "wav"|"mp3"|...}}``
      document → Text-based docs (md, csv, txt) are read and sent as
                 text blocks. PDF is sent as a native file block
                 ``{"type": "file", "file": {...}}``. Other binary
                 formats (docx, xlsx) are dropped because this
                 endpoint only accepts PDF for file blocks.

    Returns None for unsupported content types.
    """
    content_type = part_dict.get('type')
    if content_type == 'text':
      return {'type': 'text', 'text': part_dict['text']}
    if content_type == 'image':
      if 'source' in part_dict:
        url = part_dict['source']
      else:
        url = OpenAIConnector._build_data_uri(part_dict)
      if url is None:
        return None
      return {'type': 'image_url', 'image_url': {'url': url}}
    if content_type == 'audio':
      # OpenAI expects {"type": "input_audio", "input_audio":
      #   {"data": "<base64>", "format": "wav"|"mp3"}}
      audio_data = part_dict.get('data')
      if audio_data is None and 'path' in part_dict:
        with open(part_dict['path'], 'rb') as f:
          audio_data = base64.b64encode(f.read()).decode('utf-8')
      if audio_data is None:
        return None
      mime_type = part_dict.get('media_type', '')
      audio_format = OpenAIConnector._MIME_TO_AUDIO_FORMAT.get(
          mime_type, 'wav')
      return {'type': 'input_audio', 'input_audio': {
          'data': audio_data, 'format': audio_format}}
    if content_type == 'document':
      # Text-based documents (md, csv, txt): read content as string.
      text_content = content_utils.read_text_document(part_dict)
      if text_content is not None:
        return {'type': 'text', 'text': text_content}
      # PDF: send as native file block.
      if part_dict.get('media_type') != 'application/pdf':
        return None
      data_uri = OpenAIConnector._build_data_uri(part_dict)
      if data_uri is None:
        return None
      filename = 'document'
      if 'path' in part_dict:
        filename = os.path.basename(part_dict['path'])
      return {'type': 'file', 'file': {
          'file_data': data_uri, 'filename': filename}}
    return None

  @staticmethod
  def _to_responses_part(part_dict):
    """Convert a ProxAI content block to a responses.create content part.

    Used by the ``responses.create`` endpoint.

    Type mapping:
      text     → ``{"type": "input_text", "text": "..."}``
      image    → ``{"type": "input_image", "image_url": "..."}``
      document → ``{"type": "input_file", "file_data": "...", ...}``
                 Accepts all document types natively (PDF, docx,
                 xlsx, csv, markdown, txt, etc.).

    Returns None for unsupported content types.
    """
    content_type = part_dict.get('type')
    if content_type == 'text':
      return {'type': 'input_text', 'text': part_dict['text']}
    if content_type == 'image':
      if 'source' in part_dict:
        url = part_dict['source']
      else:
        url = OpenAIConnector._build_data_uri(part_dict)
      if url is None:
        return None
      return {'type': 'input_image', 'image_url': url}
    if content_type == 'document':
      data_uri = OpenAIConnector._build_data_uri(part_dict)
      if data_uri is None:
        return None
      filename = 'document'
      if 'path' in part_dict:
        filename = os.path.basename(part_dict['path'])
      return {
          'type': 'input_file',
          'file_data': data_uri,
          'filename': filename}
    return None

  @staticmethod
  def _convert_messages(messages, converter):
    """Walk messages and convert content blocks using the given converter.

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
          part = converter(block)
          if part is not None:
            parts.append(part)
        converted.append({**message, 'content': parts})
      else:
        converted.append(message)
    return converted

  def _build_responses_input(self, chat):
    """Convert an exported chat dict to responses.create input format.

    Wraps each message in ``{"type": "message", "role": ...,
    "content": [...]}`` and converts content blocks using
    ``_to_responses_part``.
    """
    input_messages = []
    for msg in chat['messages']:
      if isinstance(msg['content'], str):
        input_messages.append({
            'type': 'message', 'role': msg['role'],
            'content': [
                {'type': 'input_text', 'text': msg['content']}]})
      elif isinstance(msg['content'], list):
        parts = []
        for block in msg['content']:
          part = self._to_responses_part(block)
          if part is not None:
            parts.append(part)
        input_messages.append({
            'type': 'message', 'role': msg['role'],
            'content': parts})
    return input_messages

  def _chat_completions_create_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.chat.completions.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[
              {'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      messages = self._convert_messages(
          query_record.chat['messages'], self._to_chat_completions_part)
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

      if query_record.parameters.n is not None:
        create = functools.partial(create, n=query_record.parameters.n)

      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create,
            reasoning_effort=query_record.parameters.thinking.value.lower())

    if query_record.output_format.type == types.OutputFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    if query_record.output_format.type == types.OutputFormatType.PYDANTIC:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)
    
    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.TEXT,
            text=response.choices[0].message.content,
        )
    ]
    if response.choices is not None and len(response.choices) > 0:
      result_record.choices = []
      for choice in response.choices:
        result_record.choices.append(
            types.ChoiceType(
                content=[
                    message_content.MessageContent(
                        type=message_content.ContentType.TEXT,
                        text=choice.message.content,
                    )
                ]
            )
        )
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _beta_chat_completions_parse_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.beta.chat.completions.parse)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(
          create, messages=[
              {'role': 'user', 'content': query_record.prompt}])

    if query_record.chat is not None:
      messages = self._convert_messages(
          query_record.chat['messages'], self._to_chat_completions_part)
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

      if query_record.parameters.n is not None:
        create = functools.partial(create, n=query_record.parameters.n)
      
      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create,
            reasoning_effort=query_record.parameters.thinking.value.lower())
      
    if query_record.output_format.type == types.OutputFormatType.PYDANTIC:
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
    if response.choices is not None and len(response.choices) > 0:
      result_record.choices = []
      for choice in response.choices:
        pydantic_content = message_content.PydanticContent(
            class_name=query_record.output_format.pydantic_class.__name__,
            class_value=query_record.output_format.pydantic_class,
            instance_value=choice.message.parsed,
        )
        result_record.choices.append(
            types.ChoiceType(
                content=[
                    message_content.MessageContent(
                        type=message_content.ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=pydantic_content,
                    )
                ]
            )
        )
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _responses_create_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.responses.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(create, input=query_record.prompt)

    if query_record.chat is not None:
      input_messages = self._build_responses_input(
          query_record.chat)
      create = functools.partial(create, input=input_messages)
      if 'system_prompt' in query_record.chat:
        create = functools.partial(
            create, instructions=query_record.chat['system_prompt'])

    if query_record.system_prompt is not None:
      create = functools.partial(
          create, instructions=query_record.system_prompt)

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_output_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.thinking is not None:
        create = functools.partial(
            create,
            reasoning={
                'effort': query_record.parameters.thinking.value.lower(),
                'summary': 'auto'})
    
    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        create = functools.partial(create, tools=[{"type": "web_search"}])
  
    if query_record.output_format.type == types.OutputFormatType.JSON:
      create = functools.partial(
          create, text={'format': {
              'type': 'json_object'
          }})

    if query_record.output_format.type == types.OutputFormatType.PYDANTIC:
      create = functools.partial(
          create, text={'format': {
              'type': 'json_object'
          }})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)
    
    parsed_response = []
    for output in response.output:
      if output.type == 'message':
        for content in output.content:
          if content.annotations and len(content.annotations) > 0:
            tool_message = message_content.MessageContent(
                type=message_content.ContentType.TOOL,
                tool_content=message_content.ToolContent(
                    name='web_search',
                    kind=message_content.ToolKind.RESULT,
                    citations=[],
                ),
            )
            for annotation in content.annotations:
              tool_message.tool_content.citations.append(
                  message_content.Citation(
                      title=annotation.title,
                      url=annotation.url))
            parsed_response.append(tool_message)
          
          parsed_response.append(message_content.MessageContent(
              type=message_content.ContentType.TEXT,
              text=content.text,
          ))
      elif output.type == 'reasoning':
        for summary in output.summary:
          if summary.type == 'summary_text':
            parsed_response.append(
                message_content.MessageContent(
                    type=message_content.ContentType.THINKING,
                    text=summary.text))

    result_record.content = parsed_response
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _images_generate_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    generate = functools.partial(self.api.images.generate)
    generate = functools.partial(generate, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      generate = functools.partial(generate, prompt=query_record.prompt)

    # TODO: Add support for other image sizes and qualities
    generate = functools.partial(
        generate,
        size="1024x1024",
        quality="standard",
        n=1)
    
    response, result_record = self._safe_provider_query(generate)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.IMAGE,
            source=response.data[0].url,
        )
    ]
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _audio_speech_create_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.audio.speech.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))
    
    if query_record.prompt is not None:
      create = functools.partial(create, input=query_record.prompt)

    # TODO: Add support for other voices
    create = functools.partial(create, voice='alloy')

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.AUDIO,
            data=response.content,
        )
    ]
    
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _videos_create_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.videos.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))
    
    if query_record.prompt is not None:
      create = functools.partial(create, prompt=query_record.prompt)
    
    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)
    
    while response.status not in ("completed", "failed"):
      time.sleep(5)
      response = self.api.videos.retrieve(response.id)
      print(
          f'Status: {response.status}, '
          f'progress: {response.progress}%, '
          f'date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    if response.status == "completed":
      video_bytes = self.api.videos.download_content(
          response.id, variant="video").content
      result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.VIDEO,
            data=video_bytes,
        )
      ]
    else:
      result_record.error = Exception(
          f"Video generation failed: {response.error}")

    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)


  ENDPOINT_EXECUTORS = {
    'chat.completions.create': '_chat_completions_create_executor',
    'beta.chat.completions.parse': '_beta_chat_completions_parse_executor',
    'responses.create': '_responses_create_executor',
    'images.generate': '_images_generate_executor',
    'audio.speech.create': '_audio_speech_create_executor',
    'videos.create': '_videos_create_executor',
  }

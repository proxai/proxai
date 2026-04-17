import functools
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types as genai_types

import proxai.connectors.provider_connector as provider_connector
import proxai.connectors.providers.gemini_mock as gemini_mock
import proxai.chat.message_content as message_content
import proxai.types as types

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
InputFormatConfigType = types.InputFormatConfigType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class GeminiConnector(provider_connector.ProviderConnector):
  """Connector for Google Gemini models."""

  def init_model(self):
    return genai.Client(api_key=self.provider_token_value_map['GEMINI_API_KEY'])

  def init_mock_model(self):
    return gemini_mock.GeminiMock()

  PROVIDER_NAME = 'gemini'

  PROVIDER_API_KEYS = ['GEMINI_API_KEY']

  ENDPOINT_PRIORITY = [
      'models.generate_content',
      'models.generate_videos',
  ]

  ENDPOINT_CONFIG = {
      'models.generate_content': FeatureConfigType(
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
              document=FeatureSupportType.SUPPORTED,
              audio=FeatureSupportType.SUPPORTED,
              video=FeatureSupportType.SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
              image=FeatureSupportType.SUPPORTED,
              audio=FeatureSupportType.SUPPORTED,
          ),
      ),
      'models.generate_videos': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          input_format=InputFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              video=FeatureSupportType.SUPPORTED,
          ),
      ),
  }
  
  def _models_generate_content_executor(
      self, query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.models.generate_content)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    contents = []
    if query_record.prompt is not None:
      contents.append(
          genai_types.Content(
              parts=[genai_types.Part(text=query_record.prompt)], role='user'
          )
      )

    config = genai_types.GenerateContentConfig()
    if query_record.chat is not None:
      for message in query_record.chat['messages']:
        role = message['role']
        if role == 'assistant':
          role = 'model'
        parts = []
        if isinstance(message['content'], str):
          parts.append(genai_types.Part(text=message['content']))
        elif isinstance(message['content'], list):
          for part in message['content']:
            if part['type'] == 'text':
              parts.append(genai_types.Part(text=part['text']))
        contents.append(genai_types.Content(parts=parts, role=role))

      if 'system_prompt' in query_record.chat:
        config.system_instruction = query_record.chat['system_prompt']

    if query_record.system_prompt is not None:
      config.system_instruction = query_record.system_prompt

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        config.max_output_tokens = query_record.parameters.max_tokens

      if query_record.parameters.temperature is not None:
        config.temperature = query_record.parameters.temperature

      if query_record.parameters.stop is not None:
        if isinstance(query_record.parameters.stop, str):
          config.stop_sequences = [query_record.parameters.stop]
        else:
          config.stop_sequences = query_record.parameters.stop

      if query_record.parameters.thinking is not None:
        if query_record.parameters.thinking.value.lower() == 'low':
          config.thinking_config = genai_types.ThinkingConfig(
              thinking_budget=1024)
        elif query_record.parameters.thinking.value.lower() == 'medium':
          config.thinking_config = genai_types.ThinkingConfig(
              thinking_budget=8192)
        elif query_record.parameters.thinking.value.lower() == 'high':
          config.thinking_config = genai_types.ThinkingConfig(
              thinking_budget=24576)

    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        config.tools = [genai_types.Tool(
            google_search=genai_types.GoogleSearch())]
    
    if query_record.response_format.type == types.ResponseFormatType.JSON:
      config.response_mime_type = 'application/json'
    
    if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      config.response_mime_type = 'application/json'
      config.response_schema = query_record.response_format.pydantic_class_json_schema

    if query_record.response_format.type == types.ResponseFormatType.IMAGE:
      config.response_modalities=['IMAGE']

    if query_record.response_format.type == types.ResponseFormatType.AUDIO:
      config.response_modalities=['AUDIO']
      config.speech_config=genai_types.SpeechConfig(
          voice_config=genai_types.VoiceConfig(
              prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                  voice_name='Kore'
              )
          )
      )

    create = functools.partial(
        create,
        config=config,
        contents=contents,
    )
  
    response, result_record = self._safe_provider_query(create)
    response: genai_types.GenerateContentResponse = response
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)

    if response.candidates:
      result_record.content = []
      for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
          continue
        for part in candidate.content.parts:
          if part.text and not part.thought:
            result_record.content.append(
                message_content.MessageContent(
                    type=message_content.ContentType.TEXT,
                    text=part.text,
                )
            )
          elif part.text and part.thought:
            result_record.content.append(
                message_content.MessageContent(
                    type=message_content.ContentType.THINKING,
                    text=part.text,
                )
            )

          if part.inline_data is not None:
            if query_record.response_format.type == types.ResponseFormatType.IMAGE:
              result_record.content.append(
                  message_content.MessageContent(
                      type=message_content.ContentType.IMAGE,
                      data=part.inline_data.data,
                  )
              )
            elif query_record.response_format.type == types.ResponseFormatType.AUDIO:
              result_record.content.append(
                  message_content.MessageContent(
                      type=message_content.ContentType.AUDIO,
                      data=part.inline_data.data,
                  )
              )

        if (candidate.grounding_metadata and
            candidate.grounding_metadata.grounding_chunks):
          citations = []
          for chunk in candidate.grounding_metadata.grounding_chunks:
            citations.append(
                message_content.Citation(
                    title=chunk.web.title,
                    url=chunk.web.uri,
                ),
            )
          if citations:
            result_record.content.append(
                message_content.MessageContent(
                    type=message_content.ContentType.TOOL,
                    tool_content=message_content.ToolContent(
                        name='web_search',
                        kind=message_content.ToolKind.RESULT,
                        citations=citations,
                    ),
                )
            )
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  def _models_generate_videos_executor(
      self, query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.models.generate_videos)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))
    if query_record.prompt is not None:
      create = functools.partial(create, prompt=query_record.prompt)
    
    response, result_record = self._safe_provider_query(create)
    response: genai_types.GenerateVideosOperation = response
    if result_record.error is not None:
      return types.ExecutorResult(result_record=result_record)
    
    while not response.done:
      time.sleep(5)
      operation_executor = functools.partial(self.api.operations.get, response)
      response, result_record = self._safe_provider_query(operation_executor)
      if result_record.error is not None:
        return types.ExecutorResult(result_record=result_record)

    generated_video = response.response.generated_videos[0]
    self.api.files.download(file=generated_video.video)
    video_bytes = generated_video.video.video_bytes

    result_record.content = [
      message_content.MessageContent(
          type=message_content.ContentType.VIDEO,
          data=video_bytes,
      )
    ]
    return types.ExecutorResult(
        result_record=result_record, raw_provider_response=response)

  ENDPOINT_EXECUTORS = {
    'models.generate_content': '_models_generate_content_executor',
    'models.generate_videos': '_models_generate_videos_executor',
  }

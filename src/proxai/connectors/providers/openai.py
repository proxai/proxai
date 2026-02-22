import functools
import time

from openai import OpenAI

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class OpenAIConnector(model_connector.ProviderModelConnector):
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
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
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
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.NOT_SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.NOT_SUPPORTED,
              json=FeatureSupportType.NOT_SUPPORTED,
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
      'responses.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          messages=FeatureSupportType.BEST_EFFORT,
          system_prompt=FeatureSupportType.SUPPORTED,
          parameters=ParameterConfigType(
              max_tokens=FeatureSupportType.SUPPORTED,
              temperature=FeatureSupportType.SUPPORTED,
              stop=FeatureSupportType.BEST_EFFORT,
          ),
          tools=ToolConfigType(
              web_search=FeatureSupportType.SUPPORTED,
          ),
          response_format=ResponseFormatConfigType(
              text=FeatureSupportType.SUPPORTED,
              json=FeatureSupportType.SUPPORTED,
              pydantic=FeatureSupportType.BEST_EFFORT,
          ),
      ),
      'images.generate': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          response_format=ResponseFormatConfigType(
              image=FeatureSupportType.SUPPORTED,
          ),
      ),
      'audio.speech.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          response_format=ResponseFormatConfigType(
              audio=FeatureSupportType.SUPPORTED,
          ),
      ),
      'videos.create': FeatureConfigType(
          prompt=FeatureSupportType.SUPPORTED,
          response_format=ResponseFormatConfigType(
              video=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

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
      create = functools.partial(create, messages=query_record.chat['messages'])

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_completion_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)

    if query_record.response_format.type == types.ResponseFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record
    
    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.TEXT,
            text=response.choices[0].message.content,
        )
    ]
    return result_record

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
      create = functools.partial(create, messages=query_record.chat['messages'])

    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_completion_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)

      if query_record.parameters.stop is not None:
        create = functools.partial(create, stop=query_record.parameters.stop)
      
    if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      create = functools.partial(
          create, response_format=query_record.response_format.pydantic_class)

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record
    
    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.PYDANTIC_INSTANCE,
            pydantic_content=message_content.PydanticContent(
                class_name=query_record.response_format.pydantic_class.__name__,
                class_value=query_record.response_format.pydantic_class,
                instance_value=response.choices[0].message.parsed,
            ),
        )
    ]
    return result_record

  def _responses_create_executor(
      self,
      query_record: types.QueryRecord) -> types.Response:
    create = functools.partial(self.api.responses.create)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    if query_record.prompt is not None:
      create = functools.partial(create, input=query_record.prompt)
    
    if query_record.system_prompt is not None:
      create = functools.partial(
          create, instructions=query_record.system_prompt)
    
    if query_record.parameters is not None:
      if query_record.parameters.max_tokens is not None:
        create = functools.partial(
            create, max_completion_tokens=query_record.parameters.max_tokens)

      if query_record.parameters.temperature is not None:
        create = functools.partial(
            create, temperature=query_record.parameters.temperature)
    
    if query_record.tools is not None:
      if types.Tools.WEB_SEARCH in query_record.tools:
        create = functools.partial(create, tools=[{"type": "web_search"}])
  
    if query_record.response_format.type == types.ResponseFormatType.JSON:
      create = functools.partial(
          create, text={'format': {
              'type': 'json_object'
          }})

    if query_record.response_format.type == types.ResponseFormatType.PYDANTIC:
      create = functools.partial(
          create, text={'format': {
              'type': 'json_object'
          }})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record
    
    parsed_response = []
    for output in response.output:
      if output.type != 'message':
        continue
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
            tool_message.tool_content.citations.append(message_content.Citation(
                title=annotation.title,
                url=annotation.url,
            ))
          parsed_response.append(tool_message)
        
        parsed_response.append(message_content.MessageContent(
            type=message_content.ContentType.TEXT,
            text=content.text,
        ))

    result_record.content = parsed_response
    return result_record

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
      return result_record

    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.IMAGE,
            source=response.data[0].url,
        )
    ]
    return result_record

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
      return result_record

    result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.AUDIO,
            data=response.content,
        )
    ]
    
    return result_record

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
      return result_record
    
    tick_count = 0
    while response.status not in ("completed", "failed"):
      time.sleep(2)
      tick_count += 1
      response = self.api.videos.retrieve(response.id)
      print(f"Status: {response.status}, progress: {response.progress}%")
      if tick_count % 10 == 0:
        print(f"Video generation progress: {response.progress}% "
              f"after {tick_count*2} seconds")
    
    if response.status == "completed":
      download = self.api.videos.download(response.id)
      result_record.content = [
        message_content.MessageContent(
            type=message_content.ContentType.VIDEO,
            source=download.url,
        )
      ]
    else:
      result_record.error = Exception(
          f"Video generation failed: {response.error}")

    return result_record


  ENDPOINT_EXECUTORS = {
    'chat.completions.create': '_chat_completions_create_executor',
    'beta.chat.completions.parse': '_beta_chat_completions_parse_executor',
    'responses.create': '_responses_create_executor',
    'images.generate': '_images_generate_executor',
    'audio.speech.create': '_audio_speech_create_executor',
    'videos.create': '_videos_create_executor',
  }

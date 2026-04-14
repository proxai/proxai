import functools

from mistralai import Mistral

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.mistral_mock as mistral_mock
import proxai.types as types
import proxai.chat.message_content as message_content

FeatureConfigType = types.FeatureConfigType
FeatureSupportType = types.FeatureSupportType
ParameterConfigType = types.ParameterConfigType
ToolConfigType = types.ToolConfigType
ResponseFormatConfigType = types.ResponseFormatConfigType


class MistralConnector(model_connector.ProviderModelConnector):
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
          response_format=ResponseFormatConfigType(
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
          response_format=ResponseFormatConfigType(
              pydantic=FeatureSupportType.SUPPORTED,
          ),
      ),
  }

  def _add_common_params(self, call, query_record: types.QueryRecord):
    """Add messages, system, and parameter kwargs to a Mistral chat call."""
    if query_record.prompt is not None:
      messages = []
      if query_record.system_prompt is not None:
        messages.append(
            {'role': 'system', 'content': query_record.system_prompt})
      messages.append({'role': 'user', 'content': query_record.prompt})
      call = functools.partial(call, messages=messages)

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

    if query_record.response_format.type == types.ResponseFormatType.JSON:
      create = functools.partial(
          create, response_format={'type': 'json_object'})

    response, result_record = self._safe_provider_query(create)
    if result_record.error is not None:
      return result_record

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
    return result_record

  def _chat_parse_executor(
      self,
      query_record: types.QueryRecord) -> types.ResultRecord:
    create = functools.partial(self.api.chat.parse)
    create = functools.partial(create, model=(
        query_record.provider_model.provider_model_identifier
    ))

    create = self._add_common_params(create, query_record)

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

    if response.choices is not None and len(response.choices) > 1:
      result_record.choices = []
      for choice in response.choices:
        result_record.choices.append(
            types.ChoiceType(
                content=[
                    message_content.MessageContent(
                        type=message_content.ContentType.PYDANTIC_INSTANCE,
                        pydantic_content=message_content.PydanticContent(
                            class_name=query_record.response_format.pydantic_class.__name__,
                            class_value=query_record.response_format.pydantic_class,
                            instance_value=choice.message.parsed,
                        ),
                    )
                ]
            )
        )
    return result_record

  ENDPOINT_EXECUTORS = {
    'chat.complete': '_chat_complete_executor',
    'chat.parse': '_chat_parse_executor',
  }

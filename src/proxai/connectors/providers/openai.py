import functools
from collections.abc import Callable
from typing import Any

from openai import OpenAI

import proxai.connectors.model_connector as model_connector
import proxai.connectors.providers.openai_mock as openai_mock
import proxai.types as types

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

    response, error, error_traceback = self._safe_provider_query(create)
    if error is not None:
      return None, error, error_traceback

    return response.choices[0].message.content, None, None

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

    response, error, error_traceback = self._safe_provider_query(create)
    if error is not None:
      return None, error, error_traceback

    return response.choices[0].message.parsed, None, None

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
    
    response, error, error_traceback = self._safe_provider_query(create)
    if error is not None:
      return None, error, error_traceback

    return response.output_text, None, None
   
  ENDPOINT_EXECUTORS = {
    'chat.completions.create': '_chat_completions_create_executor',
    'beta.chat.completions.parse': '_beta_chat_completions_parse_executor',
    'responses.create': '_responses_create_executor',
  }

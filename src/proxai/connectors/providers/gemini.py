import copy
import functools
import os
from google import genai
from google.genai import types as genai_types
import proxai.types as types
import proxai.connectors.providers.gemini_mock as gemini_mock
import proxai.connectors.model_connector as model_connector
import proxai.connectors.model_configs as model_configs


class GeminiConnector(model_connector.ProviderModelConnector):
  def get_provider_name(self):
    return 'gemini'

  def init_model(self):
    return genai.Client(
      api_key=os.environ['GEMINI_API_KEY']
    )

  def init_mock_model(self):
    return gemini_mock.GeminiMock()

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    return copy.deepcopy(query_record)

  def get_token_count(self, logging_record: types.LoggingRecord):
    # Note: This temporary implementation is not accurate.
    # Better version should be calculated from the api response or at least
    # libraries like tiktoker.
    return logging_record.query_record.max_tokens

  def get_query_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return 0

  def get_response_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return logging_record.query_record.max_tokens

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Gemini uses 'user' and 'model' as roles.  'system_instruction' is a
    # different parameter.
    contents = []
    if query_record.prompt is not None:
      contents.append(genai_types.Content(
          parts=[genai_types.Part(text=query_record.prompt)], role='user'))
    if query_record.messages is not None:
      for message in query_record.messages:
        if message['role'] == 'assistant':
          contents.append(genai_types.Content(
              parts=[genai_types.Part(text=message['content'])],
              role='model'))
        if message['role'] == 'user':
          contents.append(genai_types.Content(
              parts=[genai_types.Part(text=message['content'])],
              role='user'))

    config = genai_types.GenerateContentConfig()
    if query_record.system is not None:
      config.system_instruction = query_record.system
    if query_record.max_tokens is not None:
      config.max_output_tokens = query_record.max_tokens
    if query_record.temperature is not None:
      config.temperature = query_record.temperature
    if query_record.stop is not None:
      if isinstance(query_record.stop, str):
        config.stop_sequences = [query_record.stop]
      else:
        config.stop_sequences = query_record.stop
    response = self.api.models.generate_content(
        model=self.provider_model.provider_model_identifier,
        config=config,
        contents=contents
    )
    return response.text

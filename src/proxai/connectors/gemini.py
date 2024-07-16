import copy
import functools
import math
from typing import Union, Optional
import google.generativeai as genai
import proxai.types as types
from .gemini_mock import GeminiMock
from .model_connector import ModelConnector


class GeminiConnector(ModelConnector):
  def init_model(self):
    return genai.GenerativeModel

  def init_mock_model(self):
    return GeminiMock

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    query_record = copy.deepcopy(query_record)
    _, provider_model = query_record.model
    if (query_record.system != None
        and provider_model != types.GeminiModel.GEMINI_1_5_PRO_LATEST):
      self.feature_fail(
          query_record=query_record,
          message=(
              'System instructions are only supported for the '
              'GEMINI_1_5_PRO_LATEST model.'))
      query_record.system = None
    return query_record

  def _get_token_count(self, logging_record: types.LoggingRecord):
    # Note: This temporary implementation is not accurate.
    # Better version should be calculated from the api response or at least
    # libraries like tiktoker.
    return logging_record.query_record.max_tokens

  def _get_query_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return 0

  def _get_response_token_count(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    return logging_record.query_record.max_tokens

  def _get_estimated_cost(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    # Needs to get updated all the time.
    # This is just a temporary implementation.
    query_token_count = self._get_query_token_count(logging_record)
    response_token_count = self._get_response_token_count(logging_record)
    _, provider_model = logging_record.query_record.model
    if provider_model == types.GeminiModel.GEMINI_1_5_PRO_LATEST:
      return math.floor(query_token_count * 7.0 + response_token_count * 21.0)
    elif provider_model in [
        types.GeminiModel.GEMINI_1_0_PRO,
        types.GeminiModel.GEMINI_1_0_PRO_001,
        types.GeminiModel.GEMINI_1_0_PRO_LATEST,
        types.GeminiModel.GEMINI_1_0_PRO_VISION_LATEST,
        types.GeminiModel.GEMINI_PRO,
        types.GeminiModel.GEMINI_PRO_VISION]:
      return math.floor(query_token_count * 0.5 + response_token_count * 1.5)
    else:
      raise ValueError(f'Model not found.\n{logging_record.query_record.model}')

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    # Note: Gemini uses 'user' and 'model' as roles.  'system_instruction' is a
    # different parameter.
    contents = []
    if query_record.prompt != None:
      contents.append({'role': 'user', 'parts': query_record.prompt})
    if query_record.messages != None:
      for message in query_record.messages:
        if message['role'] == 'assistant':
          contents.append({'role': 'model', 'parts': message['content']})
        if message['role'] == 'user':
          contents.append({'role': 'user', 'parts': message['content']})
    _, provider_model = query_record.model

    if query_record.system == None:
      generate_content = self.api(model_name=provider_model).generate_content
    else:
      if provider_model == types.GeminiModel.GEMINI_1_5_PRO_LATEST:
        generate_content = self.api(
            model_name=provider_model,
            system_instruction=query_record.system).generate_content
      else:
        generate_content = self.api(model_name=provider_model).generate_content

    generation_config = genai.GenerationConfig()
    if query_record.max_tokens != None:
      generation_config.max_output_tokens = query_record.max_tokens
    if query_record.temperature != None:
      generation_config.temperature = query_record.temperature
    if query_record.stop != None:
      if isinstance(query_record.stop, str):
        query_record.stop = [query_record.stop]
      generation_config.stop_sequences = query_record.stop

    response = generate_content(
        contents=contents,
        generation_config=generation_config)
    return response.text

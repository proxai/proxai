import functools
from typing import Union, Optional
import google.generativeai as genai
import proxai.types as types
from .gemini_mock import GeminiMock
from .model_connector import ModelConnector


class GeminiConnector(ModelConnector):
  def init_model(self):
    return genai.GenerativeModel(self.provider_model)

  def init_mock_model(self):
    return GeminiMock()

  def generate_text_proc(
      self, query_record: types.QueryRecord) -> str:
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
      if provider_model == self.provider_model:
        generate_content = self.api.generate_content
      else:
        generate_content = genai.GenerativeModel(
            model_name=provider_model).generate_content
    else:
      if provider_model == types.GeminiModel.GEMINI_1_5_PRO_LATEST:
        generate_content = genai.GenerativeModel(
            model_name=provider_model,
            system_instruction=query_record.system).generate_content
      else:
        self.feature_fail(
            query_record=query_record,
            message=(
                'System instructions are only supported for the '
                'GEMINI_1_5_PRO_LATEST model.'))
        generate_content = genai.GenerativeModel(
            model_name=provider_model).generate_content

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

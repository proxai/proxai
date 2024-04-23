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
      self,
      model: types.ModelType,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = None,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None
  ) -> str:
    # Note: Gemini uses 'user' and 'model' as roles.  'system_instruction' is a
    # different parameter.
    contents = []
    if prompt != None:
      contents.append({'role': 'user', 'parts': prompt})
    if messages != None:
      for message in messages:
        if message['role'] == 'assistant':
          contents.append({'role': 'model', 'parts': message['content']})
        if message['role'] == 'user':
          contents.append({'role': 'user', 'parts': message['content']})
    _, provider_model = model

    if (provider_model == self.provider_model
        and system == None):
      generate_content = self.api.generate_content
    else:
      if system == None:
        generate_content = genai.GenerativeModel(
            model_name=provider_model).generate_content
      else:
        if provider_model != types.GeminiModel.GEMINI_1_5_PRO_LATEST:
          raise ValueError('System instructions are only supported for the '
                           'GEMINI_1_5_PRO_LATEST model.')
        generate_content = genai.GenerativeModel(
            model_name=provider_model,
            system_instruction=system).generate_content

    generation_config = genai.GenerationConfig()
    if max_tokens != None:
      generation_config.max_output_tokens = max_tokens
    if temperature != None:
      generation_config.temperature = temperature
    if stop != None:
      if isinstance(stop, str):
        stop = [stop]
      generation_config.stop_sequences = stop

    response = generate_content(
        contents=contents,
        generation_config=generation_config)
    return response.text

import copy
import functools
import os
import requests
from typing import Union, Optional
import proxai.types as types
from .hugging_face_mock import HuggingFaceMock
from .model_connector import ModelConnector


class _HuggingFaceRequest:
  def __init__(self):
    self.api_url = 'https://api-inference.huggingface.co/models/'
    self.headers = {
        'Authorization': f'Bearer {os.environ["HUGGINGFACE_API_KEY"]}'}

  def generate_content(
      self,
      prompt: str,
      model: str,
      max_tokens: Optional[int]=None,
      temperature: Optional[int]=None) -> str:
    json_input = {'inputs': prompt}
    if max_tokens != None:
      json_input['max_new_tokens'] = max_tokens
    if temperature != None:
      json_input['temperature'] = temperature
    response = requests.post(
        self.api_url + model,
        headers=self.headers,
        json=json_input)
    text = response.json()[0]['generated_text']
    if text.startswith(prompt):
      text = text[len(prompt):]
    return text


class HuggingFaceConnector(ModelConnector):
  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return HuggingFaceMock()

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    query_record = copy.deepcopy(query_record)
    if query_record.system != None:
      self.feature_fail(
          query_record=query_record,
          message='Hugging Face does not support system messages.')
      query_record.system = None
    if query_record.messages != None:
      self.feature_fail(
          query_record=query_record,
          message='Hugging Face does not support message history.')
      query_record.messages = None
    if query_record.stop != None:
      self.feature_fail(
          query_record=query_record,
          message='Hugging Face does not support stop tokens.')
      query_record.stop = None
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

  def _get_estimated_price(self, logging_record: types.LoggingRecord):
    # Note: Not implemented yet.
    # Needs to get updated all the time.
    # This is just a temporary implementation.
    query_token_count = self._get_query_token_count(logging_record)
    response_token_count = self._get_response_token_count(logging_record)
    _, provider_model = logging_record.query_record.model
    if provider_model in [
        types.HuggingFaceModel.GOOGLE_GEMMA_7B_IT,
        types.HuggingFaceModel.MISTRAL_MIXTRAL_8X7B_INSTRUCT,
        types.HuggingFaceModel.MISTRAL_MISTRAL_7B_INSTRUCT,
        types.HuggingFaceModel.NOUS_HERMES_2_MIXTRAL_8X7B,
        types.HuggingFaceModel.OPENCHAT_3_5]:
      return 0
    else:
      raise ValueError(f'Model not found.\n{logging_record.query_record.model}')

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    _, provider_model = query_record.model
    create = functools.partial(
        self.api.generate_content,
        model=provider_model)
    if query_record.max_tokens != None:
      # Note: Hugging Face uses max_new_tokens instead of max_tokens.
      # This implies that input tokens are not counted.
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    completion = create(query_record.prompt)
    return completion

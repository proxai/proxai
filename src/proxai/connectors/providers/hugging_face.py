import copy
import functools
import os
import requests
from typing import Optional
import proxai.types as types
import proxai.connectors.providers.hugging_face_mock as hugging_face_mock
import proxai.connectors.model_connector as model_connector


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


class HuggingFaceConnector(model_connector.ProviderModelConnector):
  def init_model(self):
    return _HuggingFaceRequest()

  def init_mock_model(self):
    return hugging_face_mock.HuggingFaceMock()

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

  def generate_text_proc(self, query_record: types.QueryRecord) -> str:
    provider_model = query_record.provider_model
    create = functools.partial(
        self.api.generate_content,
        model=provider_model.model)
    if query_record.max_tokens != None:
      # Note: Hugging Face uses max_new_tokens instead of max_tokens.
      # This implies that input tokens are not counted.
      create = functools.partial(create, max_tokens=query_record.max_tokens)
    if query_record.temperature != None:
      create = functools.partial(create, temperature=query_record.temperature)
    completion = create(query_record.prompt)
    return completion

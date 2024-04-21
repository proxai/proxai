import datetime
from typing import Any, Dict, Optional
import proxai.types as types
from proxai.logging.utils import log_query_record


class ModelConnector(object):
  model: Optional[types.ModelType] = None
  provider: Optional[str] = None
  provider_model: Optional[str] = None
  run_type: types.RunType
  _api: Optional[Any] = None
  _logging_options: Optional[Dict] = None

  def __init__(
      self,
      model: types.ModelType,
      run_type: types.RunType,
      logging_options: Optional[dict] = None):
    self.model = model
    self.provider, self.provider_model = model
    self.run_type = run_type
    if logging_options:
      self._logging_options = logging_options

  @property
  def api(self):
    if not self._api:
      if self.run_type == types.RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  def init_model(self):
    raise NotImplementedError

  def init_mock_model(self):
    raise NotImplementedError

  def _get_query_record(
      self,
      call_type: types.CallType,
      prompt: Optional[str] = None,
      max_tokens: Optional[int] = None,
      response: Optional[str] = None,
      error: Optional[str] = None,
      start_time: Optional[datetime.datetime] = None,
      end_time: Optional[datetime.datetime] = None):
    query_record = types.QueryRecord(
        call_type=call_type,
        provider=self.provider,
        provider_model=self.provider_model)
    if prompt:
      query_record.prompt = prompt
    if max_tokens:
      query_record.max_tokens = max_tokens
    if response:
      query_record.response = response
    if error:
      query_record.error = error
    if start_time:
      query_record.start_time = start_time
    if end_time:
      query_record.end_time = end_time
    if start_time and end_time:
      query_record.response_time = end_time - start_time
    return query_record

  def generate_text(self, prompt: str, max_tokens: int) -> str:
    start_time = datetime.datetime.now()
    try:
      response =  self.generate_text_proc(prompt, max_tokens)
      query_record = self._get_query_record(
          call_type=types.CallType.GENERATE_TEXT,
          prompt=prompt,
          max_tokens=max_tokens,
          response=response,
          start_time=start_time,
          end_time=datetime.datetime.now())
      log_query_record(
          logging_options=self._logging_options, query_record=query_record)
      return response
    except Exception as e:
      query_record = self._get_query_record(
          call_type=types.CallType.GENERATE_TEXT,
          prompt=prompt,
          max_tokens=max_tokens,
          error=str(e),
          start_time=start_time,
          end_time=datetime.datetime.now())
      log_query_record(
          logging_options=self._logging_options, query_record=query_record)
      raise e


  def generate_text_proc(self, prompt: str, max_tokens: int) -> dict:
    raise NotImplementedError

import datetime
import functools
from typing import Any, Dict, Optional
import proxai.types as types
from proxai.logging.utils import log_query_record
import proxai.caching.query_cache as query_cache


class ModelConnector(object):
  model: Optional[types.ModelType] = None
  provider: Optional[str] = None
  provider_model: Optional[str] = None
  run_type: types.RunType
  query_cache_manager: Optional[query_cache.QueryCacheManager] = None
  _api: Optional[Any] = None
  _logging_options: Optional[Dict] = None

  def __init__(
      self,
      model: types.ModelType,
      run_type: types.RunType,
      logging_options: Optional[dict] = None,
      query_cache_manager: Optional[query_cache.QueryCacheManager] = None):
    self.model = model
    self.provider, self.provider_model = model
    self.run_type = run_type
    if logging_options:
      self._logging_options = logging_options
    if query_cache_manager:
      self.query_cache_manager = query_cache_manager

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

  def generate_text(
      self,
      prompt: str,
      max_tokens: int,
      use_cache: bool = True) -> str:
    start_time = datetime.datetime.now()
    query_record = types.QueryRecord(
        call_type=types.CallType.GENERATE_TEXT,
        provider=self.provider,
        provider_model=self.provider_model,
        prompt=prompt,
        max_tokens=max_tokens)

    if self.query_cache_manager and use_cache:
      # NOT READY! What to do for error? Raise exception?
      try:
        response_record = self.query_cache_manager.look(query_record)
        if response_record:
          log_query_record(
              logging_options=self._logging_options,
              query_record=query_record,
              response_record=response_record,
              from_cache=True)
          return response_record.response
      except Exception as e:
        pass

    response, error = None, None
    try:
      response = self.generate_text_proc(prompt, max_tokens)
    except Exception as e:
      error = e

    if response != None:
      query_response_record = functools.partial(
          types.QueryResponseRecord,
          response=response)
    else:
      query_response_record = functools.partial(
          types.QueryResponseRecord,
          error=str(error))
    response_record = query_response_record(
        start_time=start_time,
        end_time=datetime.datetime.now(),
        response_time=datetime.datetime.now() - start_time)

    if self.query_cache_manager and use_cache:
      self.query_cache_manager.cache(
          query_record=query_record,
          response_record=response_record)

    log_query_record(
        logging_options=self._logging_options,
        query_record=query_record,
        response_record=response_record)
    if response != None:
      return response
    raise error


  def generate_text_proc(self, prompt: str, max_tokens: int) -> dict:
    raise NotImplementedError

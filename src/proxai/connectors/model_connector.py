import datetime
import functools
from typing import Any, Dict, Optional
import proxai.types as types
from proxai.logging.utils import log_query_record, log_message
import proxai.caching.query_cache as query_cache
import proxai.type_utils as type_utils


class ModelConnector(object):
  model: Optional[types.ModelType] = None
  provider: Optional[str] = None
  provider_model: Optional[str] = None
  run_type: types.RunType
  strict_feature_test: bool = False
  query_cache_manager: Optional[query_cache.QueryCacheManager] = None
  _api: Optional[Any] = None
  _logging_options: Optional[Dict] = None

  def __init__(
      self,
      model: types.ModelType,
      run_type: types.RunType,
      logging_options: Optional[dict] = None,
      strict_feature_test: bool = False,
      query_cache_manager: Optional[query_cache.QueryCacheManager] = None):
    self.model = model
    self.provider, self.provider_model = model
    self.run_type = run_type
    self.strict_feature_test = strict_feature_test
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

  def feature_fail(
      self,
      message: str,
      query_record: Optional[types.QueryRecord] = None):
    if self.strict_feature_test:
      log_message(
          type=types.LoggingType.ERROR,
          logging_options=self._logging_options,
          query_record=query_record,
          message=message)
      raise Exception(message)
    else:
      log_message(
          type=types.LoggingType.WARNING,
          logging_options=self._logging_options,
          query_record=query_record,
          message=message)

  def generate_text(
      self,
      prompt: Optional[str] = None,
      system: Optional[str] = None,
      messages: Optional[types.MessagesType] = None,
      max_tokens: Optional[int] = 100,
      temperature: Optional[float] = None,
      stop: Optional[types.StopType] = None,
      model: Optional[types.ModelType] = None,
      use_cache: bool = True) -> str:
    if prompt != None and messages != None:
      raise ValueError('prompt and messages cannot be set at the same time.')
    if messages != None:
      type_utils.check_messages_type(messages)

    query_model = self.model
    if model != None:
      provider, _ = model
      if provider != self.provider:
        raise ValueError(
            'Model provider does not match the connector provider.')
      query_model = model

    start_time = datetime.datetime.now()
    query_record = types.QueryRecord(
        call_type=types.CallType.GENERATE_TEXT,
        model=query_model,
        prompt=prompt,
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop)

    query_record = self.feature_check(query_record=query_record)

    if self.query_cache_manager and use_cache:
      response_record = None
      try:
        response_record = self.query_cache_manager.look(query_record)
      except Exception as e:
        pass
      if response_record:
        log_query_record(
            logging_options=self._logging_options,
            query_record=query_record,
            response_record=response_record,
            from_cache=True)
        if response_record.error:
          raise Exception(response_record.error)
        elif response_record.response != None:
          return response_record.response

    response, error = None, None
    try:
      response = self.generate_text_proc(query_record=query_record)
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

  def feature_check(self, query_record: types.QueryRecord) -> types.QueryRecord:
    raise NotImplementedError

  def generate_text_proc(self, query_record: types.QueryRecord) -> dict:
    raise NotImplementedError

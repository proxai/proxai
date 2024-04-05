from datetime import datetime
from typing import Any, Dict, Optional
from proxai.types import ModelSignature, RunType
from proxai.logging.utils import log_generate_text


class ModelConnector(object):
  model_signature: Optional[ModelSignature] = None
  run_type: RunType
  _api: Optional[Any] = None
  _logging_options: Optional[Dict] = None

  def __init__(
      self,
      signature: ModelSignature,
      run_type: RunType,
      logging_options: Optional[dict] = None):
    self.model_signature = signature
    self.run_type = run_type
    if logging_options:
      self._logging_options = logging_options

  @property
  def api(self):
    if not self._api:
      if self.run_type == RunType.PRODUCTION:
        self._api = self.init_model()
      else:
        self._api = self.init_mock_model()
    return self._api

  def init_model(self):
    raise NotImplementedError

  def init_mock_model(self):
    raise NotImplementedError

  def _log_generate_text(
      self,
      start_time: datetime,
      end_time: datetime,
      prompt: Optional[str] = None,
      response: Optional[str] = None,
      error: Optional[str] = None):
    if self._logging_options:
      log_generate_text(
          logging_options=self._logging_options,
          provider=self.model_signature.provider,
          model=self.model_signature.model,
          start_time=start_time,
          end_time=end_time,
          prompt=prompt,
          response=response,
          error=error)

  def generate_text(self, prompt: str, max_tokens: int) -> str:
    start_time = datetime.now()
    try:
      response =  self.generate_text_proc(prompt, max_tokens)
    except Exception as e:
      self._log_generate_text(
          start_time=start_time,
          end_time=datetime.now(),
          prompt=prompt,
          error=str(e))
      raise e

    self._log_generate_text(
        start_time=start_time,
        end_time=datetime.now(),
        prompt=prompt,
        response=response)
    return response

  def generate_text_proc(self, prompt: str, max_tokens: int) -> dict:
    raise NotImplementedError

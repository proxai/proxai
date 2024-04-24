from typing import Dict, List, Optional


class _MockResponse(object):
  message: str

  def __init__(self):
    self.message = 'mock response'


class DatabricksMock(object):
  def create(
      self,
      model: str,
      messages: List[Dict],
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None,
      stop: Optional[List[str]]=None) -> _MockResponse:
    return _MockResponse()

from typing import Dict, List


class _MockResponse(object):
  message: str

  def __init__(self):
    self.message = 'mock response'


class DatabricksMock(object):
  def create(self, model: str, messages: List[Dict]) -> _MockResponse:
    return _MockResponse()

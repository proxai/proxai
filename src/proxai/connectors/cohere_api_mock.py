from typing import Dict, List


class _MockResponse(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class CohereMock(object):
  def chat(self, message: str, model: str) -> _MockResponse:
    return _MockResponse()

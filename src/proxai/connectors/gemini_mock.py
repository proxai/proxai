from typing import Dict, List


class _MockResponse(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class GeminiMock(object):
  def generate_content(self, prompt: str) -> _MockResponse:
    return _MockResponse()

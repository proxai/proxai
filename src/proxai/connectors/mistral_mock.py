from typing import Any, Dict, List


class _MockMessage(object):
  content: str


class _MockChoice(object):
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()
    self.message.content = 'mock response'


class _MockResponse(object):
  choices: List[_MockChoice]

  def __init__(self):
    self.choices = [_MockChoice()]


class MistralMock(object):
  def chat(self, model: str, messages: List[Any]) -> _MockResponse:
    return _MockResponse()

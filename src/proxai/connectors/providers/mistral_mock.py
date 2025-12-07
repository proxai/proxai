from typing import Any, List, Optional


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


class MockChat(object):
  def complete(
        self,
        model: str,
        messages: List[Any],
        max_tokens: Optional[int]=None,
        temperature: Optional[float]=None,
        stop: Optional[List[str]]=None) -> _MockResponse:
      return _MockResponse()

class MistralMock(object):
  chat: MockChat

  def __init__(self):
    self.chat = MockChat()

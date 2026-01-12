from typing import Any


class _MockMessage:
  content: str


class _MockChoice:
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()
    self.message.content = 'mock response'


class _MockResponse:
  choices: list[_MockChoice]

  def __init__(self):
    self.choices = [_MockChoice()]


class MockChat:
  def complete(
        self,
        model: str,
        messages: list[Any],
        max_tokens: int | None=None,
        temperature: float | None=None,
        stop: list[str] | None=None) -> _MockResponse:
      return _MockResponse()

class MistralMock:
  chat: MockChat

  def __init__(self):
    self.chat = MockChat()

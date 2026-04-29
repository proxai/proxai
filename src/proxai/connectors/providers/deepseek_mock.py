from __future__ import annotations

class _MockMessage:
  content: str
  reasoning_content: object

  def __init__(self):
    self.content = 'mock response'
    self.reasoning_content = None


class _MockChoice:
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()


class _MockResponse:
  choices: list

  def __init__(self):
    self.choices = [_MockChoice()]


class _MockCompletions:

  def create(self, **kwargs) -> _MockResponse:
    return _MockResponse()


class _MockChat:
  completions: _MockCompletions

  def __init__(self):
    self.completions = _MockCompletions()


class DeepSeekMock:
  """Mock DeepSeek API client for testing."""

  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()

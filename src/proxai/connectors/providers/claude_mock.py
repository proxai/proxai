from __future__ import annotations

# [ContentBlock(text="Hello! How can I assist you?", type='text')]


class _MockContentBlock:
  text: str
  type: str

  def __init__(self):
    self.text = 'mock response'
    self.type = 'text'


class _MockResponse:
  content: list[_MockContentBlock]
  parsed_output: any

  def __init__(self):
    self.content = [_MockContentBlock()]
    self.parsed_output = None


class _MockStream:

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False

  def get_final_message(self) -> _MockResponse:
    return _MockResponse()


class _MockMessages:

  def stream(self, **kwargs) -> _MockStream:
    return _MockStream()


class _MockBeta:
  messages: _MockMessages

  def __init__(self):
    self.messages = _MockMessages()


class ClaudeMock:
  """Mock Claude API client for testing."""

  beta: _MockBeta

  def __init__(self):
    self.beta = _MockBeta()

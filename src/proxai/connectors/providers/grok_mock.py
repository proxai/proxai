from __future__ import annotations

class _MockResponse:
  def __init__(self):
    self.content = 'mock response'
    self.reasoning_content = ''
    self.citations = []
    self.inline_citations = []


class _MockChat:
  def append(self, *args, **kwargs):
    pass

  def sample(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(self, shape, *args, **kwargs):
    return (_MockResponse(), None)


class _MockChatNamespace:
  def create(self, **kwargs) -> _MockChat:
    return _MockChat()


class GrokMock:
  """Mock xAI Grok API client for testing."""

  def __init__(self):
    self.chat = _MockChatNamespace()

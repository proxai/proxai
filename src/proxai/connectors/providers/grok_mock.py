from typing import Dict, List, Optional


class _MockResponse(object):
  content: str

  def __init__(self):
    self.content = 'mock response'


class _MockContinuedChat(object):
  def append(*args, **kwargs):
    pass

  def sample(*args, **kwargs):
    return _MockResponse()

  def parse(*args, **kwargs):
    return _MockResponse()


class _MockChat(object):
  def create(
      self,
      **kwargs) -> _MockResponse:
    return _MockContinuedChat()


class GrokMock(object):
  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()

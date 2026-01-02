from typing import Dict, List, Optional
# [ContentBlock(text="Hello! It's nice to meet you. How can I assist you today?", type='text')]

class _MockContentBlock(object):
  text: str
  type: str

  def __init__(self):
    self.text = 'mock response'
    self.type = 'text'


class _MockResponse(object):
  content: List[_MockContentBlock]

  def __init__(self):
    self.content = [_MockContentBlock()]


class _MockMessages(object):
  def create(
      self,
      **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(
      self,
      **kwargs) -> _MockResponse:
    return _MockResponse()


class _MockBeta(object):
  messages: _MockMessages

  def __init__(self):
    self.messages = _MockMessages()


class ClaudeMock(object):
  messages: _MockMessages
  beta: _MockBeta

  def __init__(self):
    self.messages = _MockMessages()
    self.beta = _MockBeta()

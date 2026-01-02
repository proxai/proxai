from typing import Dict, List, Optional


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


class _MockCompletions(object):
  def create(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()


class _MockChat(object):
  completions: _MockCompletions

  def __init__(self):
    self.completions = _MockCompletions()


class _MockBeta(object):
  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()


class DatabricksMock(object):
  chat: _MockChat
  beta: _MockBeta

  def __init__(self):
    self.chat = _MockChat()
    self.beta = _MockBeta()

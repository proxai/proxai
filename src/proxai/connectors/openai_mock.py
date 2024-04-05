from typing import Dict, List


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
  def create(self, model: str, messages: List[Dict]) -> _MockResponse:
    return _MockResponse()


class _MockChat(object):
  completions: _MockCompletions

  def __init__(self):
    self.completions = _MockCompletions()


class OpenAIMock(object):
  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()

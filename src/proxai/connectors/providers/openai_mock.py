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


class _MockResponsesResponse(object):
  output_text: str
  output_parsed: dict

  def __init__(self):
    self.output_text = 'mock response'
    self.output_parsed = {'mock': 'response'}


class _MockResponses(object):
  def create(
      self,
      model: str,
      input: str) -> _MockResponsesResponse:
    return _MockResponsesResponse()


class OpenAIMock(object):
  chat: _MockChat
  beta: _MockBeta
  responses: _MockResponses

  def __init__(self):
    self.chat = _MockChat()
    self.beta = _MockBeta()
    self.responses = _MockResponses()

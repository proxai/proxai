# [ContentBlock(text="Hello! How can I assist you?", type='text')]

class _MockContentBlock:
  text: str
  type: str

  def __init__(self):
    self.text = 'mock response'
    self.type = 'text'


class _MockResponse:
  content: list[_MockContentBlock]

  def __init__(self):
    self.content = [_MockContentBlock()]


class _MockMessages:
  def create(
      self,
      **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(
      self,
      **kwargs) -> _MockResponse:
    return _MockResponse()


class _MockBeta:
  messages: _MockMessages

  def __init__(self):
    self.messages = _MockMessages()


class ClaudeMock:
  messages: _MockMessages
  beta: _MockBeta

  def __init__(self):
    self.messages = _MockMessages()
    self.beta = _MockBeta()

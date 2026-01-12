

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


class _MockCompletions:
  def create(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()


class _MockChat:
  completions: _MockCompletions

  def __init__(self):
    self.completions = _MockCompletions()


class _MockBeta:
  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()


class DatabricksMock:
  """Mock Databricks API client for testing."""

  chat: _MockChat
  beta: _MockBeta

  def __init__(self):
    self.chat = _MockChat()
    self.beta = _MockBeta()

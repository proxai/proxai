class _MockMessage:
  content: str
  parsed: object

  def __init__(self):
    self.content = 'mock response'
    self.parsed = None


class _MockChoice:
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()


class _MockResponse:
  choices: list

  def __init__(self):
    self.choices = [_MockChoice()]


class _MockChat:
  """Mock Mistral chat surface (complete + parse)."""

  def complete(self, **kwargs) -> _MockResponse:
    return _MockResponse()

  def parse(self, **kwargs) -> _MockResponse:
    return _MockResponse()


class MistralMock:
  """Mock Mistral API client for testing."""

  def __init__(self):
    self.chat = _MockChat()

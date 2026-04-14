class _MockTextItem:
  def __init__(self):
    self.type = 'text'
    self.text = 'mock response'


class _MockMessage:
  def __init__(self):
    self.content = [_MockTextItem()]


class _MockResponse:
  def __init__(self):
    self.message = _MockMessage()


class CohereMock:
  """Mock Cohere V2 client for testing."""

  def chat(self, **kwargs) -> _MockResponse:
    return _MockResponse()

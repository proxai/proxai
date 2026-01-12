

class _MockResponse:
  content: str

  def __init__(self):
    self.content = 'mock response'


class _MockContinuedChat:
  def append(*args, **kwargs):
    pass

  def sample(*args, **kwargs):
    return _MockResponse()

  def parse(*args, **kwargs):
    return _MockResponse()


class _MockChat:
  def create(
      self,
      **kwargs) -> _MockResponse:
    return _MockContinuedChat()


class GrokMock:
  chat: _MockChat

  def __init__(self):
    self.chat = _MockChat()

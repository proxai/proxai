

class _MockContentItem:
  text: str

  def __init__(self):
    self.text = 'mock response'


class _MockMessage:
  content: list[_MockContentItem]

  def __init__(self):
    self.content = [_MockContentItem()]


class _MockResponse:
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()


class CohereMock:
  def chat(
      self,
      model: str,
      messages: list[dict] | None=None,
      max_tokens: int | None=None,
      temperature: float | None=None,
      stop_sequences: list[str] | None=None,
      response_format: dict | None=None) -> _MockResponse:
    return _MockResponse()

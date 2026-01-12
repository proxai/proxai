

class _MockResponse:
  text: str

  def __init__(self):
    self.text = 'mock response'


class _MockModel:
  def generate_content(self, *args, **kwargs) -> _MockResponse:
    return _MockResponse()


class GeminiMock:
  """Mock Gemini API client for testing."""

  models: _MockModel

  def __init__(self):
    self.models = _MockModel()

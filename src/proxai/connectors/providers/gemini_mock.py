from typing import Optional


class _MockResponse(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class GeminiMock(object):
  def __init__(self, model_name: str, system_instruction: Optional[str]=None):
    pass

  def generate_content(
      self, contents, generation_config) -> _MockResponse:
    return _MockResponse()

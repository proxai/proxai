from typing import Optional


class _MockResponse(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class _MockModel(object):
  def generate_content(
      self,
      model,
      config,
      contents) -> _MockResponse:
    return _MockResponse()


class GeminiMock(object):
  models: _MockModel

  def __init__(self):
    self.models = _MockModel()

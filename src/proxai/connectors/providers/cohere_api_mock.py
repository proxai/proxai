from typing import Dict, List, Optional


class _MockContentItem(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class _MockMessage(object):
  content: List[_MockContentItem]

  def __init__(self):
    self.content = [_MockContentItem()]


class _MockResponse(object):
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()


class CohereMock(object):
  def chat(
      self,
      model: str,
      messages: Optional[List[Dict]]=None,
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None,
      stop_sequences: Optional[List[str]]=None,
      response_format: Optional[Dict]=None) -> _MockResponse:
    return _MockResponse()

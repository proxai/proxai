from typing import Dict, List, Optional


class _MockResponse(object):
  text: str

  def __init__(self):
    self.text = 'mock response'


class CohereMock(object):
  def chat(
      self,
      message: str,
      model: str,
      preamble: Optional[str]=None,
      chat_history: Optional[List[Dict]]=None,
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None,
      stop_sequences: Optional[List[str]]=None) -> _MockResponse:
    return _MockResponse()

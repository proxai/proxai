from typing import Optional, List, Dict


class HuggingFaceMock(object):
  def generate_content(
      self,
      messages: List[Dict[str, str]],
      model: str,
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None,
      stop: Optional[List[str]]=None) -> str:
    return 'mock response'

from typing import Optional


class HuggingFaceMock(object):
  def generate_content(
      self,
      prompt: str,
      model: str,
      max_tokens: Optional[int]=None,
      temperature: Optional[float]=None) -> str:
    return 'mock response'

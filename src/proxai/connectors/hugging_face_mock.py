from typing import Dict, List


class HuggingFaceMock(object):
  def generate_content(self, prompt: str, model: str) -> str:
    return 'mock response'

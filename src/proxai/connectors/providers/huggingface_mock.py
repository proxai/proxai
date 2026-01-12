

class HuggingFaceMock:
  def generate_content(
      self,
      messages: list[dict[str, str]],
      model: str,
      max_tokens: int | None=None,
      temperature: float | None=None,
      stop: list[str] | None=None) -> str:
    return 'mock response'

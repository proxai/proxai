class _MockMessage:
  content: str
  parsed: object

  def __init__(self):
    self.content = 'mock response'
    self.parsed = None


class _MockChoice:
  message: _MockMessage

  def __init__(self):
    self.message = _MockMessage()


class _MockChatResponse:
  choices: list

  def __init__(self):
    self.choices = [_MockChoice()]


class _MockChat:
  """Mock Mistral chat surface (complete + parse)."""

  def complete(self, **kwargs) -> _MockChatResponse:
    return _MockChatResponse()

  def parse(self, **kwargs) -> _MockChatResponse:
    return _MockChatResponse()


class _MockTextChunk:
  def __init__(self, text: str):
    self.type = 'text'
    self.text = text


class _MockMessageOutput:
  def __init__(self):
    self.type = 'message.output'
    self.role = 'assistant'
    self.content = [_MockTextChunk('mock response')]


class _MockConversationResponse:
  def __init__(self):
    self.conversation_id = 'mock_conversation_id'
    self.outputs = [_MockMessageOutput()]


class _MockConversations:
  def start(self, **kwargs) -> _MockConversationResponse:
    return _MockConversationResponse()


class _MockBeta:
  def __init__(self):
    self.conversations = _MockConversations()


class MistralMock:
  """Mock Mistral API client for testing."""

  def __init__(self):
    self.chat = _MockChat()
    self.beta = _MockBeta()

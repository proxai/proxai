"""Chat class for managing conversation sessions."""

import copy

from proxai.chat.message import Message


class Chat:
  """A conversation session containing a sequence of messages.

  Wraps a list of Message objects with a full list-like interface,
  plus an optional system prompt. Provides validation on mutation
  to ensure only valid Message objects are stored.

  Attributes:
    system_prompt: Optional system prompt string for the conversation.

  Example:
    >>> import proxai as px
    >>> chat = px.Chat(system_prompt="You are a helpful assistant.")
    >>> chat.append(px.Message(role="user", content="Hello!"))
    >>> chat.append(px.Message(role="assistant", content="Hi there!"))
    >>> print(chat)  # Chat(2 messages)
    >>> print(chat[0])  # Message(role='user', content='Hello!')
  """

  def __init__(self, messages=None, system_prompt=None):
    """Initialize a Chat session.

    Args:
      messages: Optional list of Message objects or dicts to initialize with.
      system_prompt: Optional system prompt string.
    """
    self._system_prompt = system_prompt
    self._messages = []
    if messages is not None:
      for msg in messages:
        self._messages.append(self._validate_message(msg))

  @property
  def system_prompt(self) -> str | None:
    """Get the system prompt."""
    return self._system_prompt

  @system_prompt.setter
  def system_prompt(self, value: str | None):
    """Set the system prompt."""
    if value is not None and not isinstance(value, str):
      raise TypeError(
          f"system_prompt must be a string or None, got {type(value).__name__}."
      )
    self._system_prompt = value

  def _validate_message(self, msg) -> Message:
    """Validate and return a Message object."""
    if isinstance(msg, Message):
      return msg
    if isinstance(msg, dict):
      return Message.from_dict(msg)
    raise TypeError(
        f"Expected Message or dict, got {type(msg).__name__}."
    )

  def append(self, msg) -> None:
    """Append a message to the conversation."""
    self._messages.append(self._validate_message(msg))

  def extend(self, msgs) -> None:
    """Extend the conversation with multiple messages."""
    for msg in msgs:
      self._messages.append(self._validate_message(msg))

  def insert(self, index: int, msg) -> None:
    """Insert a message at the given index."""
    self._messages.insert(index, self._validate_message(msg))

  def pop(self, index: int = -1) -> Message:
    """Remove and return the message at the given index."""
    return self._messages.pop(index)

  def clear(self) -> None:
    """Remove all messages from the conversation."""
    self._messages.clear()

  def copy(self) -> "Chat":
    """Return a deep copy of this Chat."""
    return copy.deepcopy(self)

  def __getitem__(self, index):
    if isinstance(index, slice):
      new_chat = Chat(system_prompt=self._system_prompt)
      new_chat._messages = self._messages[index]
      return new_chat
    return self._messages[index]

  def __setitem__(self, index, msg):
    self._messages[index] = self._validate_message(msg)

  def __delitem__(self, index):
    del self._messages[index]

  def __len__(self) -> int:
    return len(self._messages)

  def __iter__(self):
    return iter(self._messages)

  def __add__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    new_chat = self.copy()
    new_chat._messages.extend(other._messages)
    return new_chat

  def __iadd__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    self._messages.extend(other._messages)
    return self

  def __eq__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    return (
        self._system_prompt == other._system_prompt
        and self._messages == other._messages
    )

  def __repr__(self) -> str:
    count = len(self._messages)
    suffix = "message" if count == 1 else "messages"
    return f"Chat({count} {suffix})"

  def to_dict(self) -> dict:
    """Serialize to a dictionary."""
    result = {}
    if self._system_prompt is not None:
      result["system_prompt"] = self._system_prompt
    result["messages"] = [msg.to_dict() for msg in self._messages]
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "Chat":
    """Create a Chat from a dictionary."""
    messages = [Message.from_dict(m) for m in data.get("messages", [])]
    return cls(
        messages=messages,
        system_prompt=data.get("system_prompt"),
    )

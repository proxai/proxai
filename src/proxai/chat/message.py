"""Message dataclass for conversation messages."""

import copy
import dataclasses

from proxai.chat.message_content import MessageContent, MessageRoleType, ContentType


@dataclasses.dataclass
class Message:
  """A single message in a conversation.

  Represents one turn in a conversation, with a role (user or assistant)
  and content that can be either a simple string or a list of
  MessageContent blocks for multi-modal messages.

  Attributes:
    role: The message sender role ("user" or "assistant").
    content: Message content as a string or list of MessageContent blocks.

  Example:
    >>> import proxai as px
    >>> # Simple text message
    >>> px.Message(role="user", content="Hello!")
    >>> # Multi-modal message
    >>> px.Message(role="user", content=[
    ...     px.MessageContent(type="image", source="https://example.com/img.png"),
    ...     px.MessageContent(type="text", text="What's in this image?"),
    ... ])
  """

  role: MessageRoleType | str
  content: str | list[MessageContent | str]

  def __post_init__(self):
    if isinstance(self.role, str):
      try:
        self.role = MessageRoleType(self.role)
      except ValueError:
        raise ValueError(
            f"Invalid role: {self.role!r}. "
            f"Must be one of: {[r.value for r in MessageRoleType]}"
        )
    if isinstance(self.content, list):
      normalized = []
      for item in self.content:
        if isinstance(item, str):
          normalized.append(MessageContent(type="text", text=item))
        elif isinstance(item, MessageContent):
          normalized.append(item)
        else:
          raise TypeError(
              f"Content list items must be str or MessageContent, "
              f"got {type(item).__name__}."
          )
      self.content = normalized

  def to_dict(self) -> dict:
    """Serialize to a dictionary."""
    result = {"role": self.role.value}
    if isinstance(self.content, str):
      result["content"] = self.content
    else:
      result["content"] = [item.to_dict() for item in self.content]
    return result

  @classmethod
  def from_dict(cls, data: dict) -> "Message":
    """Create a Message from a dictionary."""
    content = data["content"]
    if isinstance(content, str):
      content = [MessageContent(type=ContentType.TEXT, text=content)]
    elif isinstance(content, list):
      content = [MessageContent.from_dict(item) for item in content]
    return cls(role=data["role"], content=content)

  def copy(self) -> "Message":
    """Return a deep copy of this Message."""
    return copy.deepcopy(self)

  def __repr__(self) -> str:
    role_str = self.role.value
    if isinstance(self.content, str):
      display = (
          self.content if len(self.content) <= 50
          else self.content[:47] + "..."
      )
      return f"Message(role='{role_str}', content='{display}')"
    else:
      return (
          f"Message(role='{role_str}', content=[{len(self.content)} items])"
      )

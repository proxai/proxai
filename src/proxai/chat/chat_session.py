"""Chat class for managing conversation sessions."""

import copy
import json
import dataclasses

from proxai.chat.message import Message
from proxai.chat.message_content import MessageContent, ContentType, MessageRoleType


@dataclasses.dataclass(init=False)
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

  _system_prompt: str | None = dataclasses.field(default=None, repr=False)
  messages: list[Message] = dataclasses.field(default_factory=list)

  @property
  def system_prompt(self) -> str | None:
    return self._system_prompt

  @system_prompt.setter
  def system_prompt(self, value):
    if value is not None and not isinstance(value, str):
      raise TypeError("system_prompt must be a string.")
    self._system_prompt = value

  def __init__(self, messages=None, system_prompt=None):
    """Initialize a Chat session.

    Args:
      messages: Optional list of Message objects or dicts to initialize with.
      system_prompt: Optional system prompt string.
    """
    self.system_prompt = system_prompt
    self.messages = []
    if messages is not None:
      for msg in messages:
        self.messages.append(self._validate_message(msg))

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
    self.messages.append(self._validate_message(msg))

  def extend(self, msgs) -> None:
    """Extend the conversation with multiple messages."""
    for msg in msgs:
      self.messages.append(self._validate_message(msg))

  def insert(self, index: int, msg) -> None:
    """Insert a message at the given index."""
    self.messages.insert(index, self._validate_message(msg))

  def pop(self, index: int = -1) -> Message:
    """Remove and return the message at the given index."""
    return self.messages.pop(index)

  def clear(self) -> None:
    """Remove all messages from the conversation."""
    self.messages.clear()

  def copy(self) -> "Chat":
    """Return a deep copy of this Chat."""
    return copy.deepcopy(self)

  def __getitem__(self, index):
    if isinstance(index, slice):
      new_chat = Chat(system_prompt=self.system_prompt)
      new_chat.messages = self.messages[index]
      return new_chat
    return self.messages[index]

  def __setitem__(self, index, msg):
    self.messages[index] = self._validate_message(msg)

  def __delitem__(self, index):
    del self.messages[index]

  def __len__(self) -> int:
    return len(self.messages)

  def __iter__(self):
    return iter(self.messages)

  def __add__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    new_chat = self.copy()
    new_chat.messages.extend(other.messages)
    return new_chat

  def __iadd__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    self.messages.extend(other.messages)
    return self

  def __eq__(self, other):
    if not isinstance(other, Chat):
      return NotImplemented
    return (
        self.system_prompt == other.system_prompt
        and self.messages == other.messages
    )

  def __repr__(self) -> str:
    count = len(self.messages)
    suffix = "message" if count == 1 else "messages"
    return f"Chat({count} {suffix})"

  def to_dict(self) -> dict:
    """Serialize to a dictionary."""
    result = {}
    if self.system_prompt is not None:
      result["system_prompt"] = self.system_prompt
    result["messages"] = [msg.to_dict() for msg in self.messages]
    return result

  def export(
      self,
      merge_consecutive_roles: bool = True,
      omit_thinking: bool = True,
      allowed_types: list[ContentType | str] | None = None,
      add_json_guidance_to_system: bool = False,
      add_json_schema_guidance_to_system: dict | str | None = None,
      add_json_guidance_to_user_prompt: bool = False,
      add_json_schema_guidance_to_user_prompt: dict | str | None = None,
      add_system_to_messages: bool = False,
      add_system_to_first_user_message: bool = False,
      export_single_prompt: bool = False,
  ) -> dict | str:
    """Export chat as a dict or single prompt string.

    Processing order:
      1. Validate params and normalize allowed_types (fail fast).
      2. Build system prompt (append JSON guidance if requested).
      3. Deep copy messages.
      4. Place system prompt (into messages or first user message).
      5. Append JSON guidance to last user message (if requested).
      6. Filter thinking content.
      7. Validate allowed types.
      8. Merge consecutive same-role messages.
      9. Build result (dict or single prompt string).

    Args:
      merge_consecutive_roles: If True, consecutive messages with the same
        role are merged by extending the first message's content with the
        second message's content items.
      omit_thinking: If True, thinking content blocks are removed. Messages
        that become empty after removal are dropped.
      allowed_types: If set, only these content types are allowed. Raises
        ValueError if any message contains a disallowed type. Applied after
        omit_thinking filtering.
      add_json_guidance_to_system: If True, appends JSON response guidance
        to the system prompt. Mutually exclusive with
        add_json_schema_guidance_to_system.
      add_json_schema_guidance_to_system: If set, appends JSON schema
        guidance to the system prompt. Accepts a dict (json-serialized) or
        a pre-formatted string. Mutually exclusive with
        add_json_guidance_to_system.
      add_json_guidance_to_user_prompt: If True, appends JSON response
        guidance to the last user message. Mutually exclusive with
        add_json_schema_guidance_to_user_prompt.
      add_json_schema_guidance_to_user_prompt: If set, appends JSON schema
        guidance to the last user message. Accepts a dict or pre-formatted
        string. Mutually exclusive with add_json_guidance_to_user_prompt.
      add_system_to_messages: If True, emits the system prompt as a
        {"role": "system"} message at the start instead of a separate
        "system_prompt" field. Mutually exclusive with
        add_system_to_first_user_message.
      add_system_to_first_user_message: If True, prepends the system prompt
        to the first user message's text content. If no user message exists,
        one is created. Mutually exclusive with add_system_to_messages.
      export_single_prompt: If True, returns a single string formatted as
        "ROLE:\\ncontent" blocks separated by double newlines. Only text
        and source fields are supported; raises ValueError if any content
        uses data or path.

    Returns:
      A dict or str representation of the chat with transformations applied.
    """
    # 1. Validate params and normalize (fail fast, before deep copy).
    self._validate_export_params(
        add_json_guidance_to_system, add_json_schema_guidance_to_system,
        add_json_guidance_to_user_prompt, add_json_schema_guidance_to_user_prompt,
        add_system_to_messages, add_system_to_first_user_message,
    )
    allowed_set = self._normalize_allowed_types(allowed_types)

    # 2. Build system prompt with JSON guidance.
    system_prompt = self._build_system_prompt(
        self.system_prompt,
        add_json_guidance_to_system,
        add_json_schema_guidance_to_system,
    )

    # 3. Deep copy messages.
    messages = [msg.copy() for msg in self.messages]

    # 4. Place system prompt into messages if requested.
    if system_prompt is not None and add_system_to_first_user_message:
      self._prepend_system_to_first_user(messages, system_prompt)
      system_prompt = None

    # 5. Append JSON guidance to last user message.
    if add_json_guidance_to_user_prompt or (
        add_json_schema_guidance_to_user_prompt is not None
    ):
      self._append_guidance_to_last_user(
          messages,
          add_json_guidance_to_user_prompt,
          add_json_schema_guidance_to_user_prompt,
      )

    # 6-8. Process messages.
    if omit_thinking:
      messages = self._filter_thinking(messages)
    if allowed_set is not None:
      self._validate_allowed_types(messages, allowed_set)
    if merge_consecutive_roles:
      messages = self._merge_consecutive(messages)

    # 9. Build result.
    if export_single_prompt:
      self._validate_single_prompt_content(messages)
      return self._format_as_single_prompt(system_prompt, messages)
    return self._build_result_dict(
        system_prompt, messages, add_system_to_messages,
    )

  @staticmethod
  def _validate_export_params(
      add_json_guidance_system, add_json_schema_guidance_system,
      add_json_guidance_user, add_json_schema_guidance_user,
      add_system_to_messages, add_system_to_first_user,
  ):
    """Validate mutually exclusive export parameters."""
    if add_json_guidance_system and add_json_schema_guidance_system is not None:
      raise ValueError(
          "add_json_guidance_to_system and "
          "add_json_schema_guidance_to_system are mutually exclusive."
      )
    if add_json_guidance_user and add_json_schema_guidance_user is not None:
      raise ValueError(
          "add_json_guidance_to_user_prompt and "
          "add_json_schema_guidance_to_user_prompt are mutually exclusive."
      )
    if add_system_to_messages and add_system_to_first_user:
      raise ValueError(
          "add_system_to_messages and "
          "add_system_to_first_user_message are mutually exclusive."
      )

  @staticmethod
  def _build_json_guidance_text(
      add_json_guidance: bool,
      add_json_schema_guidance: dict | str | None,
  ) -> str | None:
    """Build JSON guidance text from parameters."""
    if add_json_guidance:
      return "You must respond with valid JSON."
    if add_json_schema_guidance is not None:
      schema_str = (
          json.dumps(add_json_schema_guidance, indent=2)
          if isinstance(add_json_schema_guidance, dict)
          else add_json_schema_guidance
      )
      return (
          "You must respond with valid JSON that follows this schema:\n"
          f"{schema_str}"
      )
    return None

  @classmethod
  def _build_system_prompt(
      cls,
      system_prompt: str | None,
      add_json_guidance: bool,
      add_json_schema_guidance: dict | str | None,
  ) -> str | None:
    """Build system prompt, appending JSON guidance if requested."""
    suffix = cls._build_json_guidance_text(
        add_json_guidance, add_json_schema_guidance,
    )
    if suffix is None:
      return system_prompt
    if system_prompt is None:
      return suffix
    return f"{system_prompt}\n\n{suffix}"

  @classmethod
  def _append_guidance_to_last_user(
      cls,
      messages: list[Message],
      add_json_guidance: bool,
      add_json_schema_guidance: dict | str | None,
  ):
    """Append JSON guidance to the last user message's content."""
    suffix = cls._build_json_guidance_text(
        add_json_guidance, add_json_schema_guidance,
    )
    if suffix is None:
      return
    for msg in reversed(messages):
      if msg.role == MessageRoleType.USER:
        if isinstance(msg.content, str):
          msg.content = f"{msg.content}\n\n{suffix}"
        elif msg.content and msg.content[-1].type == ContentType.TEXT:
          msg.content[-1].text = f"{msg.content[-1].text}\n\n{suffix}"
        else:
          msg.content.append(MessageContent(type="text", text=suffix))
        return
    messages.append(Message(role="user", content=suffix))

  @staticmethod
  def _prepend_system_to_first_user(
      messages: list[Message], system_prompt: str,
  ):
    """Prepend system prompt text to the first user message."""
    for msg in messages:
      if msg.role == MessageRoleType.USER:
        if isinstance(msg.content, str):
          msg.content = f"{system_prompt}\n\n{msg.content}"
        elif msg.content and msg.content[0].type == ContentType.TEXT:
          msg.content[0].text = (
              f"{system_prompt}\n\n{msg.content[0].text}"
          )
        else:
          msg.content.insert(
              0, MessageContent(type="text", text=system_prompt)
          )
        return
    messages.insert(0, Message(role="user", content=system_prompt))

  @staticmethod
  def _filter_thinking(messages: list[Message]) -> list[Message]:
    """Remove thinking content blocks, dropping empty messages."""
    filtered = []
    for msg in messages:
      if isinstance(msg.content, list):
        msg.content = [
            c for c in msg.content
            if c.type != ContentType.THINKING
        ]
        if msg.content:
          filtered.append(msg)
      else:
        filtered.append(msg)
    return filtered

  @staticmethod
  def _normalize_allowed_types(
      allowed_types: list[ContentType | str] | None,
  ) -> set[ContentType] | None:
    """Normalize allowed_types to a ContentType set, or None."""
    if allowed_types is None:
      return None
    allowed_set = set()
    for t in allowed_types:
      if isinstance(t, str):
        allowed_set.add(ContentType(t))
      elif isinstance(t, ContentType):
        allowed_set.add(t)
      else:
        raise TypeError(
            f"allowed_types items must be str or ContentType, "
            f"got {type(t).__name__}."
        )
    return allowed_set

  @staticmethod
  def _validate_allowed_types(
      messages: list[Message],
      allowed_set: set[ContentType],
  ):
    """Raise ValueError if any content type is not in allowed_set."""
    for msg in messages:
      if isinstance(msg.content, str):
        if ContentType.TEXT not in allowed_set:
          raise ValueError(
              f"Content type 'text' is not in allowed_types."
          )
      else:
        for c in msg.content:
          if c.type not in allowed_set:
            raise ValueError(
                f"Content type '{c.type.value}' is not in allowed_types."
            )

  @staticmethod
  def _merge_consecutive(messages: list[Message]) -> list[Message]:
    """Merge consecutive messages with the same role."""
    if not messages:
      return messages
    merged = [messages[0]]
    for i in range(1, len(messages)):
      msg = messages[i]
      last = merged[-1]
      if msg.role == last.role:
        if isinstance(last.content, str):
          last.content = [MessageContent(type="text", text=last.content)]
        if isinstance(msg.content, str):
          last.content.append(MessageContent(type="text", text=msg.content))
        else:
          last.content.extend(msg.content)
      else:
        merged.append(msg)
    return merged

  @staticmethod
  def _validate_single_prompt_content(messages: list[Message]):
    """Raise ValueError if any content uses data or path fields."""
    for msg in messages:
      if not isinstance(msg.content, list):
        continue
      for c in msg.content:
        if c.data is not None:
          raise ValueError(
              "export_single_prompt does not support "
              "'data' field in message content."
          )
        if c.path is not None:
          raise ValueError(
              "export_single_prompt does not support "
              "'path' field in message content."
          )

  @staticmethod
  def _build_result_dict(
      system_prompt: str | None,
      messages: list[Message],
      add_system_to_messages: bool,
  ) -> dict:
    """Build the export result dictionary."""
    result = {}
    exported_messages = [msg.to_dict() for msg in messages]
    if system_prompt is not None and add_system_to_messages:
      exported_messages.insert(
          0, {"role": "system", "content": system_prompt}
      )
      system_prompt = None
    if system_prompt is not None:
      result["system_prompt"] = system_prompt
    result["messages"] = exported_messages
    return result

  @staticmethod
  def _format_as_single_prompt(
      system_prompt: str | None, messages: list[Message],
  ) -> str:
    """Format system prompt and messages as a single prompt string."""
    parts = []
    if system_prompt is not None:
      parts.append(f"SYSTEM:\n{system_prompt}")
    for msg in messages:
      role_label = msg.role.value.upper()
      if isinstance(msg.content, str):
        parts.append(f"{role_label}:\n{msg.content}")
      else:
        lines = []
        for c in msg.content:
          if c.text is not None:
            lines.append(c.text)
          elif c.source is not None:
            lines.append(c.source)
        content_text = "\n".join(lines)
        parts.append(f"{role_label}:\n{content_text}")
    return "\n\n".join(parts)

  @classmethod
  def from_dict(cls, data: dict) -> "Chat":
    """Create a Chat from a dictionary."""
    messages = [Message.from_dict(m) for m in data.get("messages", [])]
    return cls(
        messages=messages,
        system_prompt=data.get("system_prompt"),
    )

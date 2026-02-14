"""Tests for Chat class."""

import pytest

from proxai.chat.chat_session import Chat
from proxai.chat.message import Message
from proxai.chat.message_content import MessageContent


def _user_msg(text):
  return Message(role="user", content=text)


def _assistant_msg(text):
  return Message(role="assistant", content=text)


class TestChatCreation:
  """Test creating Chat with different initialization options."""

  def test_empty_chat(self):
    chat = Chat()
    assert len(chat) == 0
    assert chat.system_prompt is None

  def test_chat_with_system_prompt(self):
    chat = Chat(system_prompt="You are helpful.")
    assert chat.system_prompt == "You are helpful."
    assert len(chat) == 0

  def test_chat_with_messages(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    assert len(chat) == 2
    assert chat[0].content == "Hello"
    assert chat[1].content == "Hi"

  def test_chat_from_dicts(self):
    chat = Chat(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])
    assert len(chat) == 2
    assert chat[0].content == "Hello"


class TestChatSystemPrompt:
  """Test system prompt property."""

  def test_set_and_get_system_prompt(self):
    chat = Chat()
    chat.system_prompt = "Be concise."
    assert chat.system_prompt == "Be concise."

  def test_clear_system_prompt(self):
    chat = Chat(system_prompt="Hello")
    chat.system_prompt = None
    assert chat.system_prompt is None

  def test_invalid_system_prompt_raises_error(self):
    chat = Chat()
    with pytest.raises(TypeError, match="system_prompt must be a string"):
      chat.system_prompt = 123


class TestChatMutation:
  """Test append, extend, insert, pop, clear."""

  def test_append(self):
    chat = Chat()
    chat.append(_user_msg("Hello"))
    assert len(chat) == 1
    assert chat[0].content == "Hello"

  def test_append_dict(self):
    chat = Chat()
    chat.append({"role": "user", "content": "Hello"})
    assert len(chat) == 1
    assert chat[0].content == "Hello"

  def test_append_invalid_raises_error(self):
    chat = Chat()
    with pytest.raises(TypeError, match="Expected Message or dict"):
      chat.append("not a message")

  def test_extend(self):
    chat = Chat()
    chat.extend([_user_msg("Hello"), _assistant_msg("Hi")])
    assert len(chat) == 2

  def test_insert(self):
    chat = Chat(messages=[_user_msg("First"), _user_msg("Third")])
    chat.insert(1, _user_msg("Second"))
    assert len(chat) == 3
    assert chat[1].content == "Second"

  def test_pop(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    popped = chat.pop()
    assert popped.content == "Hi"
    assert len(chat) == 1

  def test_clear(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    chat.clear()
    assert len(chat) == 0


class TestChatListInterface:
  """Test list-like operations: indexing, slicing, iteration, deletion."""

  def test_getitem(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B"), _user_msg("C")])
    assert chat[0].content == "A"
    assert chat[-1].content == "C"

  def test_slice_returns_chat(self):
    chat = Chat(
        messages=[_user_msg("A"), _user_msg("B"), _user_msg("C")],
        system_prompt="Test",
    )
    sliced = chat[1:]
    assert isinstance(sliced, Chat)
    assert len(sliced) == 2
    assert sliced[0].content == "B"
    assert sliced.system_prompt == "Test"

  def test_setitem(self):
    chat = Chat(messages=[_user_msg("Hello")])
    chat[0] = _assistant_msg("Replaced")
    assert chat[0].content == "Replaced"

  def test_delitem(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B")])
    del chat[0]
    assert len(chat) == 1
    assert chat[0].content == "B"

  def test_iter(self):
    messages = [_user_msg("A"), _assistant_msg("B")]
    chat = Chat(messages=messages)
    contents = [msg.content for msg in chat]
    assert contents == ["A", "B"]

  def test_len(self):
    chat = Chat(messages=[_user_msg("A"), _user_msg("B")])
    assert len(chat) == 2


class TestChatOperators:
  """Test __add__, __iadd__, __eq__."""

  def test_add_creates_new_chat(self):
    chat1 = Chat(messages=[_user_msg("A")], system_prompt="Sys")
    chat2 = Chat(messages=[_assistant_msg("B")])
    combined = chat1 + chat2
    assert len(combined) == 2
    assert combined.system_prompt == "Sys"
    assert len(chat1) == 1

  def test_iadd(self):
    chat1 = Chat(messages=[_user_msg("A")])
    chat2 = Chat(messages=[_assistant_msg("B")])
    chat1 += chat2
    assert len(chat1) == 2

  def test_equality(self):
    chat1 = Chat(
        messages=[_user_msg("Hello")], system_prompt="Sys"
    )
    chat2 = Chat(
        messages=[_user_msg("Hello")], system_prompt="Sys"
    )
    assert chat1 == chat2

  def test_inequality_different_messages(self):
    chat1 = Chat(messages=[_user_msg("A")])
    chat2 = Chat(messages=[_user_msg("B")])
    assert chat1 != chat2

  def test_inequality_different_system_prompt(self):
    chat1 = Chat(messages=[_user_msg("A")], system_prompt="X")
    chat2 = Chat(messages=[_user_msg("A")], system_prompt="Y")
    assert chat1 != chat2


class TestChatCopyAndRepr:
  """Test copy and repr behavior."""

  def test_copy_is_independent(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Sys")
    copied = chat.copy()
    copied.append(_assistant_msg("Hi"))
    copied.system_prompt = "Changed"
    assert len(chat) == 1
    assert chat.system_prompt == "Sys"

  def test_repr(self):
    assert repr(Chat()) == "Chat(0 messages)"
    assert repr(Chat(messages=[_user_msg("A")])) == "Chat(1 message)"
    assert repr(Chat(messages=[_user_msg("A"), _user_msg("B")])) == "Chat(2 messages)"


class TestChatSerialization:
  """Test to_dict and from_dict round-trip."""

  def test_to_dict_without_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")])
    d = chat.to_dict()
    assert "system_prompt" not in d
    assert len(d["messages"]) == 1

  def test_to_dict_with_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    d = chat.to_dict()
    assert d["system_prompt"] == "Be helpful."

  def test_round_trip(self):
    original = Chat(system_prompt="Test prompt")
    original.append(_user_msg("Hello"))
    original.append(Message(role="assistant", content=[
        MessageContent(type="text", text="Here's an image:"),
        MessageContent(type="image", source="https://example.com/img.png"),
    ]))
    restored = Chat.from_dict(original.to_dict())
    assert restored == original


class TestChatExportMergeConsecutiveRoles:
  """Test export with merge_consecutive_roles."""

  def test_merges_consecutive_same_role_strings(self):
    chat = Chat(messages=[_user_msg("Hello"), _user_msg("World")])
    result = chat.export()
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert len(result["messages"][0]["content"]) == 2
    assert result["messages"][0]["content"][0] == {"type": "text", "text": "Hello"}
    assert result["messages"][0]["content"][1] == {"type": "text", "text": "World"}

  def test_merges_consecutive_same_role_lists(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="text", text="Look at this:"),
            MessageContent(type="image", source="https://example.com/a.png"),
        ]),
        Message(role="user", content=[
            MessageContent(type="text", text="And this:"),
            MessageContent(type="image", source="https://example.com/b.png"),
        ]),
    ])
    result = chat.export()
    assert len(result["messages"]) == 1
    assert len(result["messages"][0]["content"]) == 4

  def test_does_not_merge_alternating_roles(self):
    chat = Chat(messages=[
        _user_msg("Hello"), _assistant_msg("Hi"), _user_msg("Bye"),
    ])
    result = chat.export()
    assert len(result["messages"]) == 3

  def test_merge_disabled(self):
    chat = Chat(messages=[_user_msg("Hello"), _user_msg("World")])
    result = chat.export(merge_consecutive_roles=False)
    assert len(result["messages"]) == 2

  def test_does_not_mutate_original(self):
    chat = Chat(messages=[_user_msg("Hello"), _user_msg("World")])
    chat.export()
    assert len(chat) == 2
    assert chat[0].content == "Hello"


class TestChatExportOmitThinking:
  """Test export with omit_thinking."""

  def test_removes_thinking_content_from_list(self):
    chat = Chat(messages=[
        Message(role="assistant", content=[
            MessageContent(type="thinking", text="Reasoning..."),
            MessageContent(type="text", text="Answer"),
        ]),
    ])
    result = chat.export()
    assert len(result["messages"]) == 1
    assert len(result["messages"][0]["content"]) == 1
    assert result["messages"][0]["content"][0]["type"] == "text"

  def test_drops_message_if_only_thinking(self):
    chat = Chat(messages=[
        _user_msg("Hello"),
        Message(role="assistant", content=[
            MessageContent(type="thinking", text="Reasoning..."),
        ]),
        _assistant_msg("Final answer"),
    ])
    result = chat.export()
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "Hello"

  def test_keeps_thinking_when_disabled(self):
    chat = Chat(messages=[
        Message(role="assistant", content=[
            MessageContent(type="thinking", text="Reasoning..."),
            MessageContent(type="text", text="Answer"),
        ]),
    ])
    result = chat.export(omit_thinking=False)
    assert len(result["messages"][0]["content"]) == 2

  def test_string_content_unaffected(self):
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export()
    assert result["messages"][0]["content"] == "Hello"


class TestChatExportAllowedTypes:
  """Test export with allowed_types."""

  def test_passes_when_all_types_allowed(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="text", text="Look"),
            MessageContent(type="image", source="https://example.com/img.png"),
        ]),
    ])
    result = chat.export(allowed_types=["text", "image"])
    assert len(result["messages"]) == 1

  def test_raises_on_disallowed_type(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="text", text="Look"),
            MessageContent(type="image", source="https://example.com/img.png"),
        ]),
    ])
    with pytest.raises(ValueError, match="'image' is not in allowed_types"):
      chat.export(allowed_types=["text"])

  def test_raises_on_string_content_when_text_not_allowed(self):
    chat = Chat(messages=[_user_msg("Hello")])
    with pytest.raises(ValueError, match="'text' is not in allowed_types"):
      chat.export(allowed_types=["image"])

  def test_thinking_omitted_before_allowed_types_check(self):
    chat = Chat(messages=[
        Message(role="assistant", content=[
            MessageContent(type="thinking", text="Reasoning..."),
            MessageContent(type="text", text="Answer"),
        ]),
    ])
    result = chat.export(allowed_types=["text"])
    assert len(result["messages"][0]["content"]) == 1

  def test_accepts_content_type_enum(self):
    from proxai.types import ContentType
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export(allowed_types=[ContentType.TEXT])
    assert len(result["messages"]) == 1

  def test_none_allows_everything(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="text", text="Look"),
            MessageContent(type="image", source="https://example.com/img.png"),
            MessageContent(type="audio", source="https://example.com/a.mp3"),
        ]),
    ])
    result = chat.export(allowed_types=None)
    assert len(result["messages"][0]["content"]) == 3


class TestChatExportJsonGuidance:
  """Test export with add_json_guidance_to_system."""

  def test_appends_to_existing_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(add_json_guidance_to_system=True)
    assert result["system_prompt"] == (
        "Be helpful.\n\nYou must respond with valid JSON."
    )

  def test_creates_system_prompt_when_none(self):
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export(add_json_guidance_to_system=True)
    assert result["system_prompt"] == "You must respond with valid JSON."

  def test_disabled_by_default(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export()
    assert result["system_prompt"] == "Be helpful."


class TestChatExportJsonSchemaGuidance:
  """Test export with add_json_schema_guidance_to_system."""

  def test_appends_schema_dict_to_system_prompt(self):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(add_json_schema_guidance_to_system=schema)
    assert "You must respond with valid JSON that follows this schema:" in (
        result["system_prompt"]
    )
    assert '"type": "object"' in result["system_prompt"]

  def test_appends_schema_string_to_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(
        add_json_schema_guidance_to_system='{"type": "object"}'
    )
    assert result["system_prompt"].endswith('{"type": "object"}')

  def test_creates_system_prompt_when_none(self):
    schema = {"type": "object"}
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export(add_json_schema_guidance_to_system=schema)
    assert result["system_prompt"].startswith(
        "You must respond with valid JSON"
    )

  def test_mutually_exclusive_with_json_guidance(self):
    chat = Chat(messages=[_user_msg("Hello")])
    with pytest.raises(ValueError, match="mutually exclusive"):
      chat.export(
          add_json_guidance_to_system=True,
          add_json_schema_guidance_to_system={"type": "object"},
      )


class TestChatExportSystemToMessages:
  """Test export with add_system_to_messages."""

  def test_adds_system_role_message(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(add_system_to_messages=True)
    assert "system_prompt" not in result
    assert len(result["messages"]) == 2
    assert result["messages"][0] == {
        "role": "system", "content": "Be helpful."
    }
    assert result["messages"][1]["role"] == "user"

  def test_no_system_message_when_prompt_is_none(self):
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export(add_system_to_messages=True)
    assert "system_prompt" not in result
    assert len(result["messages"]) == 1

  def test_with_json_guidance(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(
        add_system_to_messages=True, add_json_guidance_to_system=True,
    )
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"].endswith(
        "You must respond with valid JSON."
    )


class TestChatExportSystemToFirstUser:
  """Test export with add_system_to_first_user_message."""

  def test_prepends_to_string_content(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(add_system_to_first_user_message=True)
    assert "system_prompt" not in result
    assert result["messages"][0]["content"] == "Be helpful.\n\nHello"

  def test_prepends_to_first_text_in_list(self):
    chat = Chat(
        messages=[Message(role="user", content=[
            MessageContent(type="text", text="Describe this:"),
            MessageContent(type="image", source="https://example.com/a.png"),
        ])],
        system_prompt="Be helpful.",
    )
    result = chat.export(add_system_to_first_user_message=True)
    assert "system_prompt" not in result
    first_content = result["messages"][0]["content"][0]
    assert first_content["text"] == "Be helpful.\n\nDescribe this:"

  def test_inserts_text_when_first_item_is_not_text(self):
    chat = Chat(
        messages=[Message(role="user", content=[
            MessageContent(type="image", source="https://example.com/a.png"),
        ])],
        system_prompt="Be helpful.",
    )
    result = chat.export(add_system_to_first_user_message=True)
    assert "system_prompt" not in result
    content = result["messages"][0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Be helpful."}
    assert content[1]["type"] == "image"

  def test_creates_user_message_when_none_exists(self):
    chat = Chat(
        messages=[_assistant_msg("Hi")], system_prompt="Be helpful.",
    )
    result = chat.export(add_system_to_first_user_message=True)
    assert "system_prompt" not in result
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Be helpful."
    assert result["messages"][1]["role"] == "assistant"

  def test_no_change_when_prompt_is_none(self):
    chat = Chat(messages=[_user_msg("Hello")])
    result = chat.export(add_system_to_first_user_message=True)
    assert "system_prompt" not in result
    assert result["messages"][0]["content"] == "Hello"

  def test_mutually_exclusive_with_system_to_messages(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    with pytest.raises(ValueError, match="mutually exclusive"):
      chat.export(
          add_system_to_messages=True,
          add_system_to_first_user_message=True,
      )


class TestChatExportCombined:
  """Test export with multiple options interacting."""

  def test_omit_thinking_then_merge(self):
    chat = Chat(messages=[
        Message(role="assistant", content=[
            MessageContent(type="thinking", text="Hmm..."),
        ]),
        _assistant_msg("Answer"),
    ])
    result = chat.export()
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "Answer"

  def test_preserves_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export()
    assert result["system_prompt"] == "Be helpful."

  def test_empty_chat(self):
    chat = Chat()
    result = chat.export()
    assert result == {"messages": []}

  def test_json_guidance_with_system_to_first_user(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    result = chat.export(
        add_json_guidance_to_system=True,
        add_system_to_first_user_message=True,
    )
    assert "system_prompt" not in result
    assert "You must respond with valid JSON." in (
        result["messages"][0]["content"]
    )

  def test_does_not_mutate_original_system_prompt(self):
    chat = Chat(messages=[_user_msg("Hello")], system_prompt="Be helpful.")
    chat.export(add_json_guidance_to_system=True)
    assert chat.system_prompt == "Be helpful."


class TestChatExportJsonGuidanceToUserPrompt:
  """Test export with add_json_guidance_to_user_prompt."""

  def test_appends_to_last_user_string_content(self):
    chat = Chat(messages=[
        _user_msg("Hello"), _assistant_msg("Hi"), _user_msg("Tell me"),
    ])
    result = chat.export(add_json_guidance_to_user_prompt=True)
    last_msg = result["messages"][-1]
    assert last_msg["content"] == (
        "Tell me\n\nYou must respond with valid JSON."
    )

  def test_appends_to_last_text_in_list_content(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="image", source="https://example.com/a.png"),
            MessageContent(type="text", text="Describe this"),
        ]),
    ])
    result = chat.export(add_json_guidance_to_user_prompt=True)
    last_content = result["messages"][0]["content"][-1]
    assert last_content["text"] == (
        "Describe this\n\nYou must respond with valid JSON."
    )

  def test_adds_text_when_last_item_is_not_text(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="image", source="https://example.com/a.png"),
        ]),
    ])
    result = chat.export(add_json_guidance_to_user_prompt=True)
    content = result["messages"][0]["content"]
    assert len(content) == 2
    assert content[-1] == {
        "type": "text", "text": "You must respond with valid JSON."
    }

  def test_creates_user_message_when_none_exists(self):
    chat = Chat(messages=[_assistant_msg("Hi")])
    result = chat.export(add_json_guidance_to_user_prompt=True)
    assert len(result["messages"]) == 2
    assert result["messages"][-1]["role"] == "user"
    assert result["messages"][-1]["content"] == (
        "You must respond with valid JSON."
    )

  def test_does_not_mutate_original(self):
    chat = Chat(messages=[_user_msg("Hello")])
    chat.export(add_json_guidance_to_user_prompt=True)
    assert chat[0].content == "Hello"


class TestChatExportJsonSchemaGuidanceToUserPrompt:
  """Test export with add_json_schema_guidance_to_user_prompt."""

  def test_appends_schema_to_last_user_message(self):
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    chat = Chat(messages=[_user_msg("Give me data")])
    result = chat.export(add_json_schema_guidance_to_user_prompt=schema)
    content = result["messages"][0]["content"]
    assert "You must respond with valid JSON that follows this schema:" in content
    assert '"type": "object"' in content

  def test_accepts_string_schema(self):
    chat = Chat(messages=[_user_msg("Give me data")])
    result = chat.export(
        add_json_schema_guidance_to_user_prompt='{"type": "object"}'
    )
    content = result["messages"][0]["content"]
    assert content.endswith('{"type": "object"}')

  def test_mutually_exclusive_with_json_guidance(self):
    chat = Chat(messages=[_user_msg("Hello")])
    with pytest.raises(ValueError, match="mutually exclusive"):
      chat.export(
          add_json_guidance_to_user_prompt=True,
          add_json_schema_guidance_to_user_prompt={"type": "object"},
      )

  def test_can_combine_with_system_json_guidance(self):
    chat = Chat(
        messages=[_user_msg("Hello")], system_prompt="Be helpful.",
    )
    result = chat.export(
        add_json_guidance_to_system=True,
        add_json_schema_guidance_to_user_prompt={"type": "object"},
    )
    assert "You must respond with valid JSON." in result["system_prompt"]
    assert "follows this schema:" in result["messages"][0]["content"]


class TestChatExportSinglePrompt:
  """Test export with export_single_prompt."""

  def test_simple_conversation(self):
    chat = Chat(messages=[_user_msg("Hello"), _assistant_msg("Hi")])
    result = chat.export(export_single_prompt=True)
    assert result == "USER:\nHello\n\nASSISTANT:\nHi"

  def test_includes_system_prompt(self):
    chat = Chat(
        messages=[_user_msg("Hello")], system_prompt="Be helpful.",
    )
    result = chat.export(export_single_prompt=True)
    assert result == "SYSTEM:\nBe helpful.\n\nUSER:\nHello"

  def test_list_content_with_text_and_source(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="text", text="Describe this:"),
            MessageContent(type="image", source="https://example.com/a.png"),
        ]),
    ])
    result = chat.export(export_single_prompt=True)
    assert result == (
        "USER:\nDescribe this:\nhttps://example.com/a.png"
    )

  def test_raises_on_data_field(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="image", data="base64data",
                           media_type="image/png"),
        ]),
    ])
    with pytest.raises(ValueError, match="'data' field"):
      chat.export(export_single_prompt=True)

  def test_raises_on_path_field(self):
    chat = Chat(messages=[
        Message(role="user", content=[
            MessageContent(type="image", path="./photo.png"),
        ]),
    ])
    with pytest.raises(ValueError, match="'path' field"):
      chat.export(export_single_prompt=True)

  def test_with_system_to_first_user(self):
    chat = Chat(
        messages=[_user_msg("Hello")], system_prompt="Be helpful.",
    )
    result = chat.export(
        export_single_prompt=True, add_system_to_first_user_message=True,
    )
    assert result == "USER:\nBe helpful.\n\nHello"
    assert "SYSTEM:" not in result

  def test_empty_chat(self):
    result = Chat().export(export_single_prompt=True)
    assert result == ""

  def test_empty_chat_with_system_prompt(self):
    chat = Chat(system_prompt="Be helpful.")
    result = chat.export(export_single_prompt=True)
    assert result == "SYSTEM:\nBe helpful."

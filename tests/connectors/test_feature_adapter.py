"""Tests for FeatureAdapter."""

import pytest
import pydantic

import proxai.types as types
from proxai.chat.chat_session import Chat
from proxai.chat.message import Message
from proxai.connectors.feature_adapter import FeatureAdapter

S = types.FeatureSupportType.SUPPORTED
BE = types.FeatureSupportType.BEST_EFFORT
NS = types.FeatureSupportType.NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter(
    prompt=None, messages=None, system_prompt=None,
    add_system_to_messages=None,
    temperature=None, max_tokens=None, stop=None, n=None, thinking=None,
    web_search=None,
    text=None, image=None, audio=None, video=None,
    json_fmt=None, pydantic_fmt=None, multi_modal=None,
) -> FeatureAdapter:
  """Build a FeatureAdapter with the given support levels."""
  return FeatureAdapter(
      endpoint="test_endpoint",
      endpoint_feature_config=types.FeatureConfigType(
          prompt=prompt,
          messages=messages,
          system_prompt=system_prompt,
          add_system_to_messages=add_system_to_messages,
          parameters=types.ParameterConfigType(
              temperature=temperature,
              max_tokens=max_tokens,
              stop=stop,
              n=n,
              thinking=thinking,
          ),
          tools=types.ToolConfigType(web_search=web_search),
          output_format=types.OutputFormatConfigType(
              text=text, image=image, audio=audio, video=video,
              json=json_fmt, pydantic=pydantic_fmt, multi_modal=multi_modal,
          ),
      ),
  )


def _query(
    prompt=None, chat=None, system_prompt=None,
    temperature=None, max_tokens=None, stop=None, n=None, thinking=None,
    tools=None, output_format_type=None, pydantic_class=None,
) -> types.QueryRecord:
  """Build a QueryRecord with the given values."""
  parameters = None
  if any(v is not None for v in [temperature, max_tokens, stop, n, thinking]):
    parameters = types.ParameterType(
        temperature=temperature, max_tokens=max_tokens,
        stop=stop, n=n, thinking=thinking,
    )
  output_format = None
  if output_format_type is not None:
    output_format = types.OutputFormat(
        type=output_format_type, pydantic_class=pydantic_class,
    )
  return types.QueryRecord(
      prompt=prompt, chat=chat, system_prompt=system_prompt,
      parameters=parameters, tools=tools, output_format=output_format,
  )


def _chat(system_prompt=None):
  """Build a simple Chat with one user message."""
  chat = Chat(system_prompt=system_prompt)
  chat.append(Message(role="user", content="Hello"))
  chat.append(Message(role="assistant", content="Hi"))
  return chat


class _TestModel(pydantic.BaseModel):
  name: str
  age: int


# ===================================================================
# get_query_record_support_level
# ===================================================================

class TestGetQueryRecordSupportLevelEmpty:
  """Empty query with only output_format returns SUPPORTED."""

  def test_empty_query(self):
    adapter = _adapter(text=S)
    assert adapter.get_query_record_support_level(
        _query(output_format_type=types.OutputFormatType.TEXT)) == S

  def test_no_output_format_raises(self):
    adapter = _adapter()
    with pytest.raises(ValueError, match="output_format.type.*must be set"):
      adapter.get_query_record_support_level(_query())


class TestGetQueryRecordSupportLevelPrompt:
  """Support level for prompt feature."""

  def test_supported(self):
    adapter = _adapter(prompt=S, text=S)
    assert adapter.get_query_record_support_level(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.TEXT)) == S

  def test_best_effort(self):
    adapter = _adapter(prompt=BE, text=S)
    assert adapter.get_query_record_support_level(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.TEXT)) == BE

  def test_not_supported(self):
    adapter = _adapter(prompt=NS, text=S)
    assert adapter.get_query_record_support_level(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.TEXT)) == NS

  def test_none_config_treated_as_not_supported(self):
    adapter = _adapter(prompt=None, text=S)
    assert adapter.get_query_record_support_level(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.TEXT)) == NS


class TestGetQueryRecordSupportLevelMessages:
  """Support level for chat/messages feature."""

  def test_supported(self):
    adapter = _adapter(messages=S, text=S)
    assert adapter.get_query_record_support_level(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.TEXT)) == S

  def test_best_effort(self):
    adapter = _adapter(messages=BE, text=S)
    assert adapter.get_query_record_support_level(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.TEXT)) == BE

  def test_chat_with_system_prompt_checks_both(self):
    adapter = _adapter(messages=S, system_prompt=BE, text=S)
    assert adapter.get_query_record_support_level(
        _query(chat=_chat("Be helpful"),
               output_format_type=types.OutputFormatType.TEXT)) == BE

  def test_chat_without_system_prompt_ignores_system(self):
    adapter = _adapter(messages=S, system_prompt=NS, text=S)
    assert adapter.get_query_record_support_level(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.TEXT)) == S


class TestGetQueryRecordSupportLevelSystemPrompt:
  """Support level for standalone system_prompt."""

  def test_supported(self):
    adapter = _adapter(prompt=S, system_prompt=S, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", system_prompt="Be nice",
               output_format_type=types.OutputFormatType.TEXT))
    assert result == S

  def test_best_effort_is_minimum(self):
    adapter = _adapter(prompt=S, system_prompt=BE, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", system_prompt="Be nice",
               output_format_type=types.OutputFormatType.TEXT))
    assert result == BE


class TestGetQueryRecordSupportLevelParameters:
  """Support level for parameter features."""

  def test_all_supported(self):
    adapter = _adapter(prompt=S, temperature=S, max_tokens=S, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", temperature=0.5, max_tokens=100,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == S

  def test_one_best_effort_is_minimum(self):
    adapter = _adapter(prompt=S, temperature=S, max_tokens=BE, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", temperature=0.5, max_tokens=100,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == BE

  def test_unset_params_ignored(self):
    adapter = _adapter(prompt=S, temperature=NS, max_tokens=S, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", max_tokens=100,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == S

  def test_all_param_types(self):
    adapter = _adapter(
        prompt=S, temperature=S, max_tokens=S, stop=S, n=S, thinking=S,
        text=S)
    result = adapter.get_query_record_support_level(_query(
        prompt="hi", temperature=0.5, max_tokens=100,
        stop="end", n=2, thinking=types.ThinkingType.LOW,
        output_format_type=types.OutputFormatType.TEXT))
    assert result == S


class TestGetQueryRecordSupportLevelTools:
  """Support level for tool features."""

  def test_web_search_supported(self):
    adapter = _adapter(prompt=S, web_search=S, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", tools=[types.Tools.WEB_SEARCH],
               output_format_type=types.OutputFormatType.TEXT))
    assert result == S

  def test_web_search_not_supported(self):
    adapter = _adapter(prompt=S, web_search=NS, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", tools=[types.Tools.WEB_SEARCH],
               output_format_type=types.OutputFormatType.TEXT))
    assert result == NS


class TestGetQueryRecordSupportLevelOutputFormat:
  """Support level for output format features."""

  def test_text_supported(self):
    adapter = _adapter(prompt=S, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", output_format_type=types.OutputFormatType.TEXT))
    assert result == S

  def test_json_best_effort(self):
    adapter = _adapter(prompt=S, json_fmt=BE)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", output_format_type=types.OutputFormatType.JSON))
    assert result == BE

  def test_no_output_format_raises(self):
    adapter = _adapter(prompt=S)
    with pytest.raises(ValueError, match="output_format.type.*must be set"):
      adapter.get_query_record_support_level(_query(prompt="hi"))


class TestGetQueryRecordSupportLevelMinimumAcrossFeatures:
  """Minimum across all features determines overall level."""

  def test_one_not_supported_dominates(self):
    adapter = _adapter(prompt=S, temperature=S, max_tokens=NS, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", temperature=0.5, max_tokens=100,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == NS

  def test_best_effort_below_supported(self):
    adapter = _adapter(prompt=S, temperature=BE, text=S)
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", temperature=0.5,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == BE


# ===================================================================
# adapt_query_record — validation
# ===================================================================

class TestAdaptValidation:
  """Validation checks before adaptation."""

  def test_prompt_and_chat_both_set_raises(self):
    adapter = _adapter(prompt=S, messages=S)
    with pytest.raises(ValueError, match="cannot both be set"):
      adapter.adapt_query_record(
          _query(prompt="hi", chat=_chat()))

  def test_returns_deep_copy(self):
    adapter = _adapter(prompt=S)
    original = _query(prompt="hi")
    result = adapter.adapt_query_record(original)
    assert result is not original
    assert result.prompt == original.prompt


# ===================================================================
# adapt_query_record — system_prompt (prompt path)
# ===================================================================

class TestAdaptPromptSystemPrompt:
  """System prompt handling for prompt-based queries."""

  def test_supported_keeps_system(self):
    adapter = _adapter(prompt=S, system_prompt=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice"))
    assert result.system_prompt == "Be nice"
    assert result.prompt == "hi"

  def test_best_effort_merges_into_prompt(self):
    adapter = _adapter(prompt=S, system_prompt=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice"))
    assert result.system_prompt is None
    assert result.prompt == "Be nice\n\nhi"

  def test_not_supported_raises(self):
    adapter = _adapter(prompt=S, system_prompt=NS)
    with pytest.raises(ValueError, match="system_prompt.*not supported"):
      adapter.adapt_query_record(
          _query(prompt="hi", system_prompt="Be nice"))

  def test_no_system_prompt_does_nothing(self):
    adapter = _adapter(prompt=S, system_prompt=NS)
    result = adapter.adapt_query_record(_query(prompt="hi"))
    assert result.prompt == "hi"
    assert result.system_prompt is None


# ===================================================================
# adapt_query_record — Pattern 2 (add_system_to_messages) prompt path
# ===================================================================

class TestAdaptPromptAddSystemToMessages:
  """Pattern 2: prompt + system_prompt folded into a chat-shaped messages list."""

  def test_prompt_and_system_folded_into_chat(self):
    adapter = _adapter(
        prompt=S, system_prompt=S, add_system_to_messages=True)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice"))
    assert result.prompt is None
    assert result.system_prompt is None
    assert result.chat == {
        "messages": [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "hi"},
        ],
    }

  def test_prompt_without_system_is_left_alone(self):
    adapter = _adapter(
        prompt=S, system_prompt=S, add_system_to_messages=True)
    result = adapter.adapt_query_record(_query(prompt="hi"))
    assert result.prompt == "hi"
    assert result.chat is None
    assert result.system_prompt is None

  def test_json_guidance_lands_on_user_message(self):
    adapter = _adapter(
        prompt=S, system_prompt=S, add_system_to_messages=True,
        text=S, json_fmt=BE)
    result = adapter.adapt_query_record(
        _query(
            prompt="hi", system_prompt="Be nice",
            output_format_type=types.OutputFormatType.JSON))
    assert result.chat["messages"][0] == {
        "role": "system", "content": "Be nice"}
    assert "valid JSON" in result.chat["messages"][1]["content"]
    assert result.chat["messages"][1]["content"].startswith("hi")

  def test_pydantic_schema_lands_on_user_message(self):
    adapter = _adapter(
        prompt=S, system_prompt=S, add_system_to_messages=True,
        text=S, pydantic_fmt=BE)
    result = adapter.adapt_query_record(
        _query(
            prompt="hi", system_prompt="Be nice",
            output_format_type=types.OutputFormatType.PYDANTIC,
            pydantic_class=_TestModel))
    user_content = result.chat["messages"][1]["content"]
    assert user_content.startswith("hi")
    assert "schema" in user_content.lower()
    assert "name" in user_content

  def test_best_effort_system_prompt_overrides_pattern_2(self):
    # BEST_EFFORT merges system_prompt into prompt and clears it BEFORE the
    # Pattern 2 check runs, so no chat conversion happens.
    adapter = _adapter(
        prompt=S, system_prompt=BE, add_system_to_messages=True)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice"))
    assert result.prompt == "Be nice\n\nhi"
    assert result.chat is None
    assert result.system_prompt is None


# ===================================================================
# adapt_query_record — system_prompt (chat path)
# ===================================================================

class TestAdaptChatSystemPrompt:
  """System prompt handling for chat-based queries."""

  def test_supported_keeps_system_in_export(self):
    adapter = _adapter(messages=S, system_prompt=S)
    result = adapter.adapt_query_record(
        _query(chat=_chat("Be nice")))
    assert result.chat["system_prompt"] == "Be nice"

  def test_best_effort_folds_into_first_user(self):
    adapter = _adapter(messages=S, system_prompt=BE)
    result = adapter.adapt_query_record(
        _query(chat=_chat("Be nice")))
    assert "system_prompt" not in result.chat
    first_msg = result.chat["messages"][0]
    assert "Be nice" in first_msg["content"]

  def test_not_supported_raises(self):
    adapter = _adapter(messages=S, system_prompt=NS)
    with pytest.raises(ValueError, match="system_prompt.*not supported"):
      adapter.adapt_query_record(
          _query(chat=_chat("Be nice")))

  def test_no_system_prompt_ignores_config(self):
    adapter = _adapter(messages=S, system_prompt=NS)
    result = adapter.adapt_query_record(_query(chat=_chat()))
    assert "system_prompt" not in result.chat


# ===================================================================
# adapt_query_record — messages
# ===================================================================

class TestAdaptChatMessages:
  """Messages feature handling for chat-based queries."""

  def test_supported_exports_as_dict(self):
    adapter = _adapter(messages=S)
    result = adapter.adapt_query_record(_query(chat=_chat()))
    assert isinstance(result.chat, dict)
    assert "messages" in result.chat
    assert result.prompt is None

  def test_best_effort_exports_as_single_prompt(self):
    adapter = _adapter(messages=BE)
    result = adapter.adapt_query_record(_query(chat=_chat()))
    assert result.chat is None
    assert isinstance(result.prompt, str)
    assert "USER:" in result.prompt
    assert "Hello" in result.prompt

  def test_not_supported_raises(self):
    adapter = _adapter(messages=NS)
    with pytest.raises(ValueError, match="messages.*not supported"):
      adapter.adapt_query_record(_query(chat=_chat()))

  def test_best_effort_messages_and_best_effort_system(self):
    adapter = _adapter(messages=BE, system_prompt=BE)
    result = adapter.adapt_query_record(
        _query(chat=_chat("Be nice")))
    assert result.chat is None
    assert isinstance(result.prompt, str)
    assert "Be nice" in result.prompt
    assert "Hello" in result.prompt


# ===================================================================
# adapt_query_record — parameters
# ===================================================================

class TestAdaptParameters:
  """Parameter feature handling."""

  def test_supported_keeps_value(self):
    adapter = _adapter(prompt=S, temperature=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi", temperature=0.7))
    assert result.parameters.temperature == 0.7

  def test_best_effort_removes_value(self):
    adapter = _adapter(prompt=S, temperature=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi", temperature=0.7))
    assert result.parameters is None

  def test_not_supported_raises(self):
    adapter = _adapter(prompt=S, temperature=NS)
    with pytest.raises(ValueError, match="temperature.*not supported"):
      adapter.adapt_query_record(_query(prompt="hi", temperature=0.7))

  def test_mixed_support_keeps_supported_removes_best_effort(self):
    adapter = _adapter(prompt=S, temperature=S, max_tokens=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi", temperature=0.7, max_tokens=100))
    assert result.parameters.temperature == 0.7
    assert result.parameters.max_tokens is None

  def test_all_removed_sets_parameters_to_none(self):
    adapter = _adapter(prompt=S, temperature=BE, max_tokens=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi", temperature=0.7, max_tokens=100))
    assert result.parameters is None

  def test_each_parameter_type(self):
    adapter = _adapter(
        prompt=S, temperature=S, max_tokens=S, stop=S, n=S, thinking=S)
    result = adapter.adapt_query_record(_query(
        prompt="hi", temperature=0.5, max_tokens=100,
        stop="end", n=2, thinking=types.ThinkingType.LOW))
    assert result.parameters.temperature == 0.5
    assert result.parameters.max_tokens == 100
    assert result.parameters.stop == "end"
    assert result.parameters.n == 2
    assert result.parameters.thinking == types.ThinkingType.LOW


# ===================================================================
# adapt_query_record — tools
# ===================================================================

class TestAdaptTools:
  """Tool feature handling."""

  def test_web_search_supported(self):
    adapter = _adapter(prompt=S, web_search=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi", tools=[types.Tools.WEB_SEARCH]))
    assert result.tools == [types.Tools.WEB_SEARCH]

  def test_web_search_not_supported_raises(self):
    adapter = _adapter(prompt=S, web_search=NS)
    with pytest.raises(ValueError, match="web_search.*not supported"):
      adapter.adapt_query_record(
          _query(prompt="hi", tools=[types.Tools.WEB_SEARCH]))

  def test_web_search_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, web_search=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi", tools=[types.Tools.WEB_SEARCH]))

  def test_no_tools_skips_check(self):
    adapter = _adapter(prompt=S, web_search=NS)
    result = adapter.adapt_query_record(_query(prompt="hi"))
    assert result.tools is None


# ===================================================================
# adapt_query_record — output format: text/image/audio/video
# ===================================================================

class TestAdaptOutputFormatSimple:
  """Output format handling for text, image, audio, video."""

  def test_text_supported(self):
    adapter = _adapter(prompt=S, text=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi", output_format_type=types.OutputFormatType.TEXT))
    assert result.output_format.type == types.OutputFormatType.TEXT

  def test_text_not_supported_raises(self):
    adapter = _adapter(prompt=S, text=NS)
    with pytest.raises(ValueError, match="output_format.*not supported"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.TEXT))

  def test_text_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, text=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.TEXT))

  def test_image_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, image=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.IMAGE))

  def test_audio_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, audio=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.AUDIO))

  def test_video_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, video=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.VIDEO))

  def test_multi_modal_best_effort_raises_developer_error(self):
    adapter = _adapter(prompt=S, multi_modal=BE)
    with pytest.raises(Exception, match="cannot be best effort"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.MULTI_MODAL))

  def test_no_output_format_does_nothing(self):
    adapter = _adapter(prompt=S)
    result = adapter.adapt_query_record(_query(prompt="hi"))
    assert result.output_format is None


# ===================================================================
# adapt_query_record — output format: JSON
# ===================================================================

class TestAdaptOutputFormatJSON:
  """JSON output format adds guidance to prompt or chat."""

  def test_json_supported_adds_guidance_to_prompt(self):
    adapter = _adapter(prompt=S, json_fmt=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.JSON))
    assert "You must respond with valid JSON." in result.prompt

  def test_json_best_effort_adds_guidance_to_prompt(self):
    adapter = _adapter(prompt=S, json_fmt=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.JSON))
    assert "You must respond with valid JSON." in result.prompt

  def test_json_not_supported_raises(self):
    adapter = _adapter(prompt=S, json_fmt=NS)
    with pytest.raises(ValueError, match="output_format.*not supported"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.JSON))

  def test_json_supported_adds_guidance_to_chat_export(self):
    adapter = _adapter(messages=S, json_fmt=S)
    result = adapter.adapt_query_record(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.JSON))
    system = result.chat.get("system_prompt", "")
    messages_text = str(result.chat["messages"])
    assert "valid JSON" in system or "valid JSON" in messages_text

  def test_json_with_system_prompt_best_effort_ordering(self):
    """System prompt merges first, then JSON guidance appends."""
    adapter = _adapter(prompt=S, system_prompt=BE, json_fmt=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice",
               output_format_type=types.OutputFormatType.JSON))
    assert result.system_prompt is None
    assert result.prompt.startswith("Be nice\n\nhi")
    assert result.prompt.endswith("You must respond with valid JSON.")


# ===================================================================
# adapt_query_record — output format: Pydantic
# ===================================================================

class TestAdaptOutputFormatPydantic:
  """Pydantic output format handling."""

  def test_pydantic_supported_does_nothing(self):
    adapter = _adapter(prompt=S, pydantic_fmt=S)
    result = adapter.adapt_query_record(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.PYDANTIC,
               pydantic_class=_TestModel))
    assert result.prompt == "hi"

  def test_pydantic_best_effort_adds_schema_to_prompt(self):
    adapter = _adapter(prompt=S, pydantic_fmt=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi",
               output_format_type=types.OutputFormatType.PYDANTIC,
               pydantic_class=_TestModel))
    assert "You must respond with valid JSON that follows this schema:" \
        in result.prompt
    assert '"name"' in result.prompt
    assert '"age"' in result.prompt

  def test_pydantic_best_effort_adds_schema_to_chat_export(self):
    adapter = _adapter(messages=S, pydantic_fmt=BE)
    result = adapter.adapt_query_record(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.PYDANTIC,
               pydantic_class=_TestModel))
    exported = str(result.chat)
    assert "schema" in exported.lower() or "name" in exported

  def test_pydantic_not_supported_raises(self):
    adapter = _adapter(prompt=S, pydantic_fmt=NS)
    with pytest.raises(ValueError, match="output_format.*not supported"):
      adapter.adapt_query_record(
          _query(prompt="hi",
                 output_format_type=types.OutputFormatType.PYDANTIC,
                 pydantic_class=_TestModel))

  def test_pydantic_with_system_prompt_best_effort_ordering(self):
    """System prompt merges first, then schema guidance appends."""
    adapter = _adapter(prompt=S, system_prompt=BE, pydantic_fmt=BE)
    result = adapter.adapt_query_record(
        _query(prompt="hi", system_prompt="Be nice",
               output_format_type=types.OutputFormatType.PYDANTIC,
               pydantic_class=_TestModel))
    assert result.system_prompt is None
    assert result.prompt.startswith("Be nice\n\nhi")
    assert "follows this schema:" in result.prompt


# ===================================================================
# adapt_query_record — combined scenarios
# ===================================================================

class TestAdaptCombined:
  """Combined feature adaptation scenarios."""

  def test_prompt_with_all_supported(self):
    adapter = _adapter(
        prompt=S, system_prompt=S, temperature=S, max_tokens=S,
        web_search=S, text=S)
    result = adapter.adapt_query_record(_query(
        prompt="hi", system_prompt="Be nice",
        temperature=0.7, max_tokens=100,
        tools=[types.Tools.WEB_SEARCH],
        output_format_type=types.OutputFormatType.TEXT))
    assert result.prompt == "hi"
    assert result.system_prompt == "Be nice"
    assert result.parameters.temperature == 0.7
    assert result.tools == [types.Tools.WEB_SEARCH]

  def test_chat_messages_best_effort_with_json_best_effort(self):
    adapter = _adapter(messages=BE, json_fmt=BE)
    result = adapter.adapt_query_record(
        _query(chat=_chat(),
               output_format_type=types.OutputFormatType.JSON))
    assert result.chat is None
    assert isinstance(result.prompt, str)
    assert "Hello" in result.prompt

  def test_chat_system_best_effort_messages_best_effort(self):
    """Both system and messages best-effort: system folds into user,
    then whole chat becomes a single prompt string."""
    adapter = _adapter(messages=BE, system_prompt=BE)
    result = adapter.adapt_query_record(
        _query(chat=_chat("Be nice")))
    assert result.chat is None
    assert "Be nice" in result.prompt
    assert "Hello" in result.prompt

  def test_does_not_mutate_original(self):
    adapter = _adapter(prompt=S, system_prompt=BE, temperature=BE)
    original = _query(
        prompt="hi", system_prompt="Be nice", temperature=0.7)
    adapter.adapt_query_record(original)
    assert original.prompt == "hi"
    assert original.system_prompt == "Be nice"
    assert original.parameters.temperature == 0.7


# ===================================================================
# FeatureAdapter with model_feature_config
# ===================================================================

class TestFeatureAdapterWithModelConfig:
  """Tests for FeatureAdapter with model_feature_config."""

  def test_both_none_raises(self):
    with pytest.raises(ValueError, match="At least one"):
      FeatureAdapter(endpoint="test")

  def test_only_model_config_uses_model_config(self):
    model_config = types.FeatureConfigType(prompt=S, messages=BE)
    adapter = FeatureAdapter(
        endpoint="test", model_feature_config=model_config)
    assert adapter.feature_config is model_config
    assert adapter.endpoint_feature_config is None

  def test_no_model_config_uses_endpoint_config(self):
    ep_config = types.FeatureConfigType(prompt=S, messages=BE)
    adapter = FeatureAdapter(
        endpoint="test", endpoint_feature_config=ep_config)
    assert adapter.feature_config is ep_config

  def test_model_config_merges_with_endpoint(self):
    ep_config = types.FeatureConfigType(prompt=S, messages=S)
    model_config = types.FeatureConfigType(prompt=BE, messages=S)
    adapter = FeatureAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.feature_config.prompt == BE
    assert adapter.feature_config.messages == S

  def test_model_config_restricts_support_level(self):
    ep_config = types.FeatureConfigType(
        prompt=S,
        parameters=types.ParameterConfigType(temperature=S),
        output_format=types.OutputFormatConfigType(text=S),
    )
    model_config = types.FeatureConfigType(
        prompt=S,
        parameters=types.ParameterConfigType(temperature=NS),
        output_format=types.OutputFormatConfigType(text=S),
    )
    adapter = FeatureAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    result = adapter.get_query_record_support_level(
        _query(prompt="hi", temperature=0.5,
               output_format_type=types.OutputFormatType.TEXT))
    assert result == NS

  def test_originals_stored(self):
    ep_config = types.FeatureConfigType(prompt=S)
    model_config = types.FeatureConfigType(prompt=BE)
    adapter = FeatureAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.endpoint_feature_config is ep_config
    assert adapter.model_feature_config is model_config


# ===================================================================
# get_feature_tags_support_level
# ===================================================================

class TestGetFeatureTagsSupportLevel:
  """Tests for get_feature_tags_support_level."""

  def test_empty_tags_returns_supported(self):
    adapter = _adapter(prompt=NS, text=NS)
    assert adapter.get_feature_tags_support_level([]) == S

  def test_single_supported_tag(self):
    adapter = _adapter(prompt=S, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == S

  def test_single_not_supported_tag(self):
    adapter = _adapter(prompt=NS, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == NS

  def test_single_best_effort_tag(self):
    adapter = _adapter(prompt=BE, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == BE

  def test_minimum_across_tags(self):
    adapter = _adapter(prompt=S, messages=BE, text=S)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.PROMPT,
        types.FeatureTag.MESSAGES,
    ])
    assert result == BE

  def test_not_supported_dominates(self):
    adapter = _adapter(prompt=S, messages=NS, text=S)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.PROMPT,
        types.FeatureTag.MESSAGES,
    ])
    assert result == NS

  def test_parameter_tags(self):
    adapter = _adapter(
        prompt=S, temperature=S, max_tokens=BE, stop=S, n=S, thinking=S,
        text=S)
    result = adapter.get_feature_tags_support_level([
        types.FeatureTag.TEMPERATURE,
        types.FeatureTag.MAX_TOKENS,
    ])
    assert result == BE

  def test_none_config_treated_as_not_supported(self):
    adapter = _adapter(prompt=None, text=S)
    assert adapter.get_feature_tags_support_level(
        [types.FeatureTag.PROMPT]) == NS

  def test_all_feature_tags(self):
    adapter = _adapter(
        prompt=S, messages=S, system_prompt=S,
        temperature=S, max_tokens=S, stop=S, n=S, thinking=S,
        text=S,
    )
    all_tags = list(types.FeatureTag)
    assert adapter.get_feature_tags_support_level(all_tags) == S

"""Tests for adapter_utils shared helpers."""

import pytest

import proxai.types as types
from proxai.connectors.adapter_utils import (
    min_support, merge_support_fields, merge_feature_configs,
    resolve_tag_support,
)

S = types.FeatureSupportType.SUPPORTED
BE = types.FeatureSupportType.BEST_EFFORT
NS = types.FeatureSupportType.NOT_SUPPORTED


# ===================================================================
# min_support
# ===================================================================

class TestMinSupport:
  """Tests for min_support helper."""

  def test_both_supported(self):
    assert min_support(S, S) == S

  def test_supported_and_best_effort(self):
    assert min_support(S, BE) == BE

  def test_best_effort_and_not_supported(self):
    assert min_support(BE, NS) == NS

  def test_none_treated_as_not_supported(self):
    assert min_support(S, None) == NS

  def test_both_none(self):
    assert min_support(None, None) == NS

  def test_symmetric(self):
    assert min_support(BE, S) == BE
    assert min_support(NS, S) == NS


# ===================================================================
# merge_support_fields
# ===================================================================

class TestMergeSupportFields:
  """Tests for merge_support_fields helper."""

  def test_both_none_returns_none(self):
    result = merge_support_fields(None, None, types.ParameterConfigType)
    assert result is None

  def test_one_none_uses_defaults(self):
    a = types.ParameterConfigType(temperature=S, max_tokens=BE)
    result = merge_support_fields(a, None, types.ParameterConfigType)
    assert result.temperature == NS  # S vs None(NS) -> NS
    assert result.max_tokens == NS   # BE vs None(NS) -> NS

  def test_merge_takes_minimum(self):
    a = types.ParameterConfigType(temperature=S, max_tokens=S, stop=BE)
    b = types.ParameterConfigType(temperature=BE, max_tokens=S, stop=S)
    result = merge_support_fields(a, b, types.ParameterConfigType)
    assert result.temperature == BE
    assert result.max_tokens == S
    assert result.stop == BE

  def test_tool_config(self):
    a = types.ToolConfigType(web_search=S)
    b = types.ToolConfigType(web_search=BE)
    result = merge_support_fields(a, b, types.ToolConfigType)
    assert result.web_search == BE

  def test_response_format_config(self):
    a = types.ResponseFormatConfigType(text=S, json=S, pydantic=BE)
    b = types.ResponseFormatConfigType(text=S, json=BE, pydantic=S)
    result = merge_support_fields(a, b, types.ResponseFormatConfigType)
    assert result.text == S
    assert result.json == BE
    assert result.pydantic == BE


# ===================================================================
# merge_feature_configs
# ===================================================================

class TestMergeFeatureConfigs:
  """Tests for merge_feature_configs."""

  def test_top_level_fields_take_minimum(self):
    ep = types.FeatureConfigType(prompt=S, messages=S, system_prompt=BE)
    model = types.FeatureConfigType(prompt=BE, messages=S, system_prompt=S)
    result = merge_feature_configs(ep, model)
    assert result.prompt == BE
    assert result.messages == S
    assert result.system_prompt == BE

  def test_add_system_to_messages_or_logic(self):
    ep = types.FeatureConfigType(add_system_to_messages=True)
    model = types.FeatureConfigType(add_system_to_messages=None)
    assert merge_feature_configs(ep, model).add_system_to_messages is True

    ep2 = types.FeatureConfigType(add_system_to_messages=None)
    model2 = types.FeatureConfigType(add_system_to_messages=True)
    assert merge_feature_configs(ep2, model2).add_system_to_messages is True

    ep3 = types.FeatureConfigType(add_system_to_messages=None)
    model3 = types.FeatureConfigType(add_system_to_messages=None)
    assert merge_feature_configs(ep3, model3).add_system_to_messages is None

  def test_nested_configs_merged(self):
    ep = types.FeatureConfigType(
        parameters=types.ParameterConfigType(temperature=S, max_tokens=BE),
        tools=types.ToolConfigType(web_search=S),
        response_format=types.ResponseFormatConfigType(text=S, json=S),
    )
    model = types.FeatureConfigType(
        parameters=types.ParameterConfigType(temperature=BE, max_tokens=S),
        tools=types.ToolConfigType(web_search=S),
        response_format=types.ResponseFormatConfigType(text=S, json=BE),
    )
    result = merge_feature_configs(ep, model)
    assert result.parameters.temperature == BE
    assert result.parameters.max_tokens == BE
    assert result.tools.web_search == S
    assert result.response_format.text == S
    assert result.response_format.json == BE

  def test_none_nested_configs(self):
    ep = types.FeatureConfigType(
        parameters=types.ParameterConfigType(temperature=S),
        tools=None,
    )
    model = types.FeatureConfigType(
        parameters=None,
        tools=types.ToolConfigType(web_search=S),
    )
    result = merge_feature_configs(ep, model)
    assert result.parameters.temperature == NS  # S vs None(NS) -> NS
    assert result.tools.web_search == NS         # None(NS) vs S -> NS

  def test_both_nested_none_returns_none(self):
    ep = types.FeatureConfigType(parameters=None, tools=None)
    model = types.FeatureConfigType(parameters=None, tools=None)
    result = merge_feature_configs(ep, model)
    assert result.parameters is None
    assert result.tools is None


# ===================================================================
# resolve_tag_support
# ===================================================================

class TestResolveTagSupport:
  """Tests for resolve_tag_support helper."""

  def test_top_level_prompt(self):
    config = types.FeatureConfigType(prompt=S)
    assert resolve_tag_support(config, types.FeatureTagType.PROMPT) == S

  def test_top_level_messages(self):
    config = types.FeatureConfigType(messages=BE)
    assert resolve_tag_support(config, types.FeatureTagType.MESSAGES) == BE

  def test_top_level_system_prompt(self):
    config = types.FeatureConfigType(system_prompt=NS)
    assert resolve_tag_support(
        config, types.FeatureTagType.SYSTEM_PROMPT) == NS

  def test_parameter_temperature(self):
    config = types.FeatureConfigType(
        parameters=types.ParameterConfigType(temperature=S))
    assert resolve_tag_support(
        config, types.FeatureTagType.TEMPERATURE) == S

  def test_parameter_max_tokens(self):
    config = types.FeatureConfigType(
        parameters=types.ParameterConfigType(max_tokens=BE))
    assert resolve_tag_support(
        config, types.FeatureTagType.MAX_TOKENS) == BE

  def test_parameter_none_config(self):
    config = types.FeatureConfigType(parameters=None)
    assert resolve_tag_support(
        config, types.FeatureTagType.TEMPERATURE) == NS

  def test_tool_web_search(self):
    config = types.FeatureConfigType(
        tools=types.ToolConfigType(web_search=S))
    assert resolve_tag_support(
        config, types.FeatureTagType.WEB_SEARCH) == S

  def test_tool_none_config(self):
    config = types.FeatureConfigType(tools=None)
    assert resolve_tag_support(
        config, types.FeatureTagType.WEB_SEARCH) == NS

  def test_response_format_text(self):
    config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(text=S))
    assert resolve_tag_support(
        config, types.FeatureTagType.RESPONSE_TEXT) == S

  def test_response_format_json(self):
    config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(json=BE))
    assert resolve_tag_support(
        config, types.FeatureTagType.RESPONSE_JSON) == BE

  def test_response_format_none_config(self):
    config = types.FeatureConfigType(response_format=None)
    assert resolve_tag_support(
        config, types.FeatureTagType.RESPONSE_TEXT) == NS

  def test_none_field_treated_as_not_supported(self):
    config = types.FeatureConfigType(prompt=None)
    assert resolve_tag_support(config, types.FeatureTagType.PROMPT) == NS

  def test_all_tags_covered(self):
    config = types.FeatureConfigType(
        prompt=S, messages=S, system_prompt=S,
        parameters=types.ParameterConfigType(
            temperature=S, max_tokens=S, stop=S, n=S, thinking=S),
        tools=types.ToolConfigType(web_search=S),
        response_format=types.ResponseFormatConfigType(
            text=S, image=S, audio=S, video=S,
            json=S, pydantic=S, multi_modal=S),
    )
    for tag in types.FeatureTagType:
      assert resolve_tag_support(config, tag) == S

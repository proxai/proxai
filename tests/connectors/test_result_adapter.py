"""Tests for ResultAdapter."""

import pytest

import proxai.types as types
from proxai.connectors.result_adapter import ResultAdapter

S = types.FeatureSupportType.SUPPORTED
BE = types.FeatureSupportType.BEST_EFFORT
NS = types.FeatureSupportType.NOT_SUPPORTED


# ===================================================================
# ResultAdapter init config resolution
# ===================================================================

class TestResultAdapterInitConfig:
  """Tests for ResultAdapter config resolution in __init__."""

  def test_both_none_raises(self):
    with pytest.raises(ValueError, match="At least one"):
      ResultAdapter(endpoint="test")

  def test_only_endpoint_config(self):
    ep_config = types.FeatureConfigType(prompt=S)
    adapter = ResultAdapter(
        endpoint="test", endpoint_feature_config=ep_config)
    assert adapter.feature_config is ep_config
    assert adapter.model_feature_config is None

  def test_only_model_config(self):
    model_config = types.FeatureConfigType(prompt=S)
    adapter = ResultAdapter(
        endpoint="test", model_feature_config=model_config)
    assert adapter.feature_config is model_config
    assert adapter.endpoint_feature_config is None

  def test_both_configs_merges(self):
    ep_config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(text=S, json=S))
    model_config = types.FeatureConfigType(
        response_format=types.ResponseFormatConfigType(text=S, json=BE))
    adapter = ResultAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.feature_config.response_format.json == BE
    assert adapter.feature_config.response_format.text == S

  def test_originals_stored(self):
    ep_config = types.FeatureConfigType(prompt=S)
    model_config = types.FeatureConfigType(prompt=BE)
    adapter = ResultAdapter(
        endpoint="test",
        endpoint_feature_config=ep_config,
        model_feature_config=model_config,
    )
    assert adapter.endpoint_feature_config is ep_config
    assert adapter.model_feature_config is model_config

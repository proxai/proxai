import pytest

import proxai.connections.api_key_manager as api_key_manager
import proxai.connectors.model_configs as model_configs
import proxai.types as types


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  """Remove all provider API keys from the environment."""
  for api_key_list in model_configs.PROVIDER_KEY_MAP.values():
    for api_key in api_key_list:
      monkeypatch.delenv(api_key, raising=False)
  yield


def _create_manager(
    proxdash_connection=None,
) -> api_key_manager.ApiKeyManager:
  return api_key_manager.ApiKeyManager(
      init_from_params=api_key_manager.ApiKeyManagerParams(
          proxdash_connection=proxdash_connection,
      )
  )


class TestApiKeyManagerInit:

  def test_init_no_proxdash_no_env(self):
    mgr = _create_manager()
    assert mgr.providers_with_key == {}
    assert mgr.proxdash_provider_api_keys == {}

  def test_init_picks_up_env_vars(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    mgr = _create_manager()
    assert 'openai' in mgr.providers_with_key
    assert mgr.providers_with_key['openai'] == {
        'OPENAI_API_KEY': 'test-openai-key'
    }

  def test_init_picks_up_multiple_providers(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-claude')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini')
    mgr = _create_manager()
    assert 'openai' in mgr.providers_with_key
    assert 'claude' in mgr.providers_with_key
    assert 'gemini' in mgr.providers_with_key
    assert len(mgr.providers_with_key) == 3

  def test_init_multi_key_provider(self, monkeypatch):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'test-token')
    monkeypatch.setenv('DATABRICKS_HOST', 'https://test.cloud.databricks.com')
    mgr = _create_manager()
    assert mgr.providers_with_key['databricks'] == {
        'DATABRICKS_TOKEN': 'test-token',
        'DATABRICKS_HOST': 'https://test.cloud.databricks.com',
    }

  def test_init_partial_multi_key_provider(self, monkeypatch):
    monkeypatch.setenv('DATABRICKS_TOKEN', 'test-token')
    mgr = _create_manager()
    assert mgr.providers_with_key['databricks'] == {
        'DATABRICKS_TOKEN': 'test-token',
    }

  def test_init_ignores_unknown_env_vars(self, monkeypatch):
    monkeypatch.setenv('RANDOM_API_KEY', 'random-value')
    mgr = _create_manager()
    assert mgr.providers_with_key == {}


class TestLoadProviderKeys:

  def test_reload_picks_up_new_env_vars(self, monkeypatch):
    mgr = _create_manager()
    assert mgr.providers_with_key == {}

    monkeypatch.setenv('OPENAI_API_KEY', 'new-key')
    mgr.load_provider_keys()
    assert 'openai' in mgr.providers_with_key
    assert mgr.providers_with_key['openai']['OPENAI_API_KEY'] == 'new-key'

  def test_reload_removes_deleted_env_vars(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    mgr = _create_manager()
    assert 'openai' in mgr.providers_with_key

    monkeypatch.delenv('OPENAI_API_KEY')
    mgr.load_provider_keys()
    assert 'openai' not in mgr.providers_with_key

  def test_reload_updates_changed_values(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'old-key')
    mgr = _create_manager()
    assert mgr.providers_with_key['openai']['OPENAI_API_KEY'] == 'old-key'

    monkeypatch.setenv('OPENAI_API_KEY', 'new-key')
    mgr.load_provider_keys()
    assert mgr.providers_with_key['openai']['OPENAI_API_KEY'] == 'new-key'


class TestProxDashKeyPriority:

  def test_proxdash_keys_take_priority(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    mgr = _create_manager()
    mgr.proxdash_provider_api_keys = {
        'OPENAI_API_KEY': 'proxdash-key',
    }
    mgr.load_provider_keys()
    assert mgr.providers_with_key['openai']['OPENAI_API_KEY'] == 'proxdash-key'

  def test_env_used_when_proxdash_missing(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini')
    mgr = _create_manager()
    mgr.proxdash_provider_api_keys = {
        'OPENAI_API_KEY': 'proxdash-key',
    }
    mgr.load_provider_keys()
    assert mgr.providers_with_key['openai']['OPENAI_API_KEY'] == 'proxdash-key'
    assert mgr.providers_with_key['gemini']['GEMINI_API_KEY'] == 'env-gemini'


class TestGetProviderKeys:

  def test_returns_keys_for_existing_provider(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    mgr = _create_manager()
    keys = mgr.get_provider_keys('openai')
    assert keys == {'OPENAI_API_KEY': 'test-key'}

  def test_returns_empty_dict_for_missing_provider(self):
    mgr = _create_manager()
    keys = mgr.get_provider_keys('openai')
    assert keys == {}

  def test_returns_empty_dict_for_unknown_provider(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    mgr = _create_manager()
    keys = mgr.get_provider_keys('nonexistent')
    assert keys == {}


class TestHasProviderKey:

  def test_true_when_key_exists(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    mgr = _create_manager()
    assert mgr.has_provider_key('openai') is True

  def test_false_when_key_missing(self):
    mgr = _create_manager()
    assert mgr.has_provider_key('openai') is False

  def test_false_for_unknown_provider(self):
    mgr = _create_manager()
    assert mgr.has_provider_key('nonexistent') is False


class TestStateSerialization:

  def test_state_round_trip(self, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini')
    mgr = _create_manager()

    state = mgr.get_state()
    assert isinstance(state, types.ApiKeyManagerState)
    assert 'openai' in state.providers_with_key
    assert 'gemini' in state.providers_with_key

    restored = api_key_manager.ApiKeyManager(init_from_state=state)
    assert restored.providers_with_key == mgr.providers_with_key
    assert restored.proxdash_provider_api_keys == mgr.proxdash_provider_api_keys
    assert restored.has_provider_key('openai')
    assert restored.has_provider_key('gemini')
    assert not restored.has_provider_key('claude')

  def test_empty_state_round_trip(self):
    mgr = _create_manager()
    state = mgr.get_state()
    restored = api_key_manager.ApiKeyManager(init_from_state=state)
    assert restored.providers_with_key == {}

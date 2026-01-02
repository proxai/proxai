import os
import pytest
import proxai.connectors.model_configs as model_configs

# Store original value globally
_ORIGINAL_API_KEY = None

def pytest_configure(config):
  """Configure pytest before any imports happen."""
  global _ORIGINAL_API_KEY
  _ORIGINAL_API_KEY = os.environ.pop('PROXDASH_API_KEY', None)
  pytest.model_configs_instance = model_configs.ModelConfigs()


def pytest_unconfigure(config):
  """Restore environment after all tests complete."""
  if _ORIGINAL_API_KEY is not None:
    os.environ['PROXDASH_API_KEY'] = _ORIGINAL_API_KEY

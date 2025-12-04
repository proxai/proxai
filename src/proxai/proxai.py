"""ProxAI - Unified AI Integration Platform.

This module provides backward-compatible access to ProxAI functionality.
All functions delegate to a default ProxAIClient instance.

For advanced usage (multiprocessing, multi-client), use ProxAIClient directly:
    from proxai.client import ProxAIClient
    client = ProxAIClient(cache_path="/tmp/cache")
"""

from typing import Any, Dict, Optional, Union
import proxai.types as types
from proxai.client import ProxAIClient
import proxai.stat_types as stat_types

# Re-export types for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions

# Default client instance
_DEFAULT_CLIENT: Optional[ProxAIClient] = None


def _get_default_client() -> ProxAIClient:
  """Get or create the default client."""
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    _DEFAULT_CLIENT = ProxAIClient()
  return _DEFAULT_CLIENT


def set_run_type(run_type: types.RunType):
  """Set the run type for the default client."""
  _get_default_client().run_type = run_type


def set_model(
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    generate_text: Optional[types.ProviderModelIdentifierType] = None):
  """Set the model for the default client."""
  _get_default_client().set_model(
      provider_model=provider_model,
      generate_text=generate_text)


def connect(
    experiment_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    cache_options: Optional[CacheOptions] = None,
    logging_path: Optional[str] = None,
    logging_options: Optional[LoggingOptions] = None,
    proxdash_options: Optional[ProxDashOptions] = None,
    allow_multiprocessing: Optional[bool] = True,
    model_test_timeout: Optional[int] = 25,
    strict_feature_test: Optional[bool] = False,
    suppress_provider_errors: Optional[bool] = False):
  """Configure the default ProxAI client.

  This creates a new default client with the specified configuration.
  All subsequent calls to generate_text(), etc. will use this client.
  """
  global _DEFAULT_CLIENT
  _DEFAULT_CLIENT = ProxAIClient(
      experiment_path=experiment_path,
      cache_path=cache_path,
      cache_options=cache_options,
      logging_path=logging_path,
      logging_options=logging_options,
      proxdash_options=proxdash_options,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
      strict_feature_test=strict_feature_test,
      suppress_provider_errors=suppress_provider_errors)


def generate_text(
    prompt: Optional[str] = None,
    system: Optional[str] = None,
    messages: Optional[types.MessagesType] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[types.StopType] = None,
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    use_cache: Optional[bool] = None,
    unique_response_limit: Optional[int] = None,
    extensive_return: bool = False,
    suppress_provider_errors: Optional[bool] = None) -> Union[str, types.LoggingRecord]:
  """Generate text using the default client."""
  return _get_default_client().generate_text(
      prompt=prompt,
      system=system,
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      stop=stop,
      provider_model=provider_model,
      use_cache=use_cache,
      unique_response_limit=unique_response_limit,
      extensive_return=extensive_return,
      suppress_provider_errors=suppress_provider_errors)


def get_summary(
    run_time: bool = False,
    json: bool = False) -> Union[stat_types.RunStats, Dict[str, Any]]:
  """Get summary from the default client."""
  return _get_default_client().get_summary(run_time=run_time, json=json)


def get_available_models():
  """Get available models from the default client."""
  return _get_default_client().get_available_models()


def get_current_options(
    json: bool = False) -> Union[types.RunOptions, Dict[str, Any]]:
  """Get current options from the default client."""
  return _get_default_client().get_current_options(json=json)


def reset_platform_cache():
  """Reset platform cache for the default client."""
  if _DEFAULT_CLIENT is not None:
    _DEFAULT_CLIENT.reset_platform_cache()


def reset_state():
  """Reset the default client to fresh state."""
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is not None:
    _DEFAULT_CLIENT.reset_platform_cache()
  _DEFAULT_CLIENT = None


def check_health(
    experiment_path: Optional[str] = None,
    verbose: bool = True,
    allow_multiprocessing: bool = True,
    model_test_timeout: int = 25,
    extensive_return: bool = False,
) -> types.ModelStatus:
  """Check health using the default client."""
  return _get_default_client().check_health(
      experiment_path=experiment_path,
      verbose=verbose,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
      extensive_return=extensive_return)


# === New Functions for Advanced Usage ===

def export_client_state() -> types.ProxAIClientState:
  """Export default client state for multiprocessing.

  Example:
      state = px.export_client_state()
      # Pass state to worker process

  In worker:
      px.import_client_state(state)
      px.generate_text("Hello from worker")
  """
  return _get_default_client().export_state()


def import_client_state(state: types.ProxAIClientState):
  """Import client state (typically in a worker process).

  This replaces the default client with one loaded from the given state.
  """
  global _DEFAULT_CLIENT
  _DEFAULT_CLIENT = ProxAIClient.from_state(state)


def get_client() -> ProxAIClient:
  """Get the default client instance.

  Useful for advanced operations not exposed via module functions.
  """
  return _get_default_client()

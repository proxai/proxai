from typing import Any, Dict, Optional, Union
import proxai.types as types
import proxai.client as client
import proxai.stat_types as stat_types

# Re-export for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions

_DEFAULT_CLIENT: Optional[client.ProxAIClient] = None


def _get_default_client() -> client.ProxAIClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        proxai_client_params = client.ProxAIClientParams()
        _DEFAULT_CLIENT = client.ProxAIClient(
            init_from_params=proxai_client_params)
    return _DEFAULT_CLIENT


def connect(**kwargs):
    global _DEFAULT_CLIENT
    proxai_client_params = client.ProxAIClientParams(**kwargs)
    _DEFAULT_CLIENT = client.ProxAIClient(init_from_params=proxai_client_params)


def generate_text(prompt: Optional[str] = None, **kwargs):
    return _get_default_client().generate_text(prompt=prompt, **kwargs)


# def set_model(**kwargs):
#     _get_default_client().set_model(**kwargs)


# def set_run_type(run_type: types.RunType):
#     _get_default_client().run_type = run_type


# def check_health(**kwargs):
#     return _get_default_client().check_health(**kwargs)


# def get_summary(**kwargs):
#     return _get_default_client().get_summary(**kwargs)


def get_available_models():
    return _get_default_client().available_models_instance


# def get_current_options(**kwargs):
#     return _get_default_client().get_current_options(**kwargs)


# def reset_state():
#     global _DEFAULT_CLIENT
#     if _DEFAULT_CLIENT is not None:
#         _DEFAULT_CLIENT.reset_platform_cache()
#     _DEFAULT_CLIENT = None


# def reset_platform_cache():
#     _get_default_client().reset_platform_cache()


# # === New Functions ===

# def export_client_state() -> types.ProxAIClientState:
#     """Export state for multiprocessing."""
#     return _get_default_client().export_state()


# def import_client_state(state: types.ProxAIClientState):
#     """Import state in worker process."""
#     global _DEFAULT_CLIENT
#     _DEFAULT_CLIENT = ProxAIClient.from_state(state)


# def get_client() -> ProxAIClient:
#     """Get the default client instance."""
#     return _get_default_client()

import datetime
import copy
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


def set_model(
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    generate_text: Optional[types.ProviderModelIdentifierType] = None):
  _get_default_client().set_model(
      provider_model=provider_model,
      generate_text=generate_text)


# def set_run_type(run_type: types.RunType):
#     _get_default_client().run_type = run_type


def check_health(
    experiment_path: Optional[str]=None,
    verbose: bool = True,
    allow_multiprocessing: bool = True,
    model_test_timeout: int = 25,
    extensive_return: bool = False,
):
  px_client_params = client.ProxAIClientParams(
      experiment_path=experiment_path,
      allow_multiprocessing=allow_multiprocessing,
      model_test_timeout=model_test_timeout,
  )
  px_client = client.ProxAIClient(init_from_params=px_client_params)
  if verbose:
    print('> Starting to test each model...')
  model_status = px_client.available_models_instance.list_working_models(
      clear_model_cache=True,
      verbose=verbose,
      return_all=True)
  if verbose:
    providers = set(
        [model.provider for model in model_status.working_models] +
        [model.provider for model in model_status.failed_models])
    result_table = {
        provider: {'working': [], 'failed': []} for provider in providers}
    for model in model_status.working_models:
      result_table[model.provider]['working'].append(model.model)
    for model in model_status.failed_models:
      result_table[model.provider]['failed'].append(model.model)
    print('> Finished testing.\n'
          f'   Registered Providers: {len(providers)}\n'
          f'   Succeeded Models: {len(model_status.working_models)}\n'
          f'   Failed Models: {len(model_status.failed_models)}')
    for provider in sorted(providers):
      print(f'> {provider}:')
      for model in sorted(result_table[provider]['working']):
        provider_model = px_client.model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ WORKING | {duration.total_seconds():6.2f}s ]: {model}')
      for model in sorted(result_table[provider]['failed']):
        provider_model = px_client.model_configs_instance.get_provider_model(
            (provider, model))
        duration = model_status.provider_queries[
            provider_model].response_record.response_time
        print(f'   [ FAILED  | {duration.total_seconds():6.2f}s ]: {model}')
  if extensive_return:
    return model_status



# def get_summary(**kwargs):
#     return _get_default_client().get_summary(**kwargs)


def get_available_models():
    return _get_default_client().available_models_instance


def get_current_options(json: bool = False):
    return _get_default_client().get_current_options(json=json)


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

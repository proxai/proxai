from typing import Optional
import proxai.types as types
import proxai.client as client

# Re-export for backward compatibility
CacheOptions = types.CacheOptions
LoggingOptions = types.LoggingOptions
ProxDashOptions = types.ProxDashOptions
ResponseFormat = types.StructuredResponseFormat
ResponseFormatType = types.ResponseFormatType

_DEFAULT_CLIENT: Optional[client.ProxAIClient] = None


def get_default_proxai_client() -> client.ProxAIClient:
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
  return get_default_proxai_client().generate_text(prompt=prompt, **kwargs)


def set_model(
    provider_model: Optional[types.ProviderModelIdentifierType] = None,
    generate_text: Optional[types.ProviderModelIdentifierType] = None):
  get_default_proxai_client().set_model(
      provider_model=provider_model,
      generate_text=generate_text)


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


def get_current_options(json: bool = False):
  return get_default_proxai_client().get_current_options(json=json)


def reset_state():
  global _DEFAULT_CLIENT
  if _DEFAULT_CLIENT is None:
    return
  if _DEFAULT_CLIENT.platform_used_for_default_model_cache:
    _DEFAULT_CLIENT.model_cache_manager.clear_cache()
  _DEFAULT_CLIENT = None


class DefaultModelsConnector:
  def list_models(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_models(*args, **kwargs))

  def list_providers(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_providers(*args, **kwargs))

  def list_provider_models(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_provider_models(*args, **kwargs))

  def get_model(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.get_model(*args, **kwargs))

  def list_working_models(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_models(*args, **kwargs))

  def list_working_providers(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_providers(*args, **kwargs))

  def list_working_provider_models(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.list_working_provider_models(*args, **kwargs))

  def get_working_model(self, *args, **kwargs):
    return (
        get_default_proxai_client()
        .available_models_instance.get_working_model(*args, **kwargs))

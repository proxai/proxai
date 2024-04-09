import proxai.types as types


def check_model_type(model: types.ModelType):
  """Check if model type is supported."""
  provider, provider_model = model

  providers = set(item.value for item in types.Provider)
  if provider not in providers:
    raise ValueError(
      f'Provider not supported: {provider}. Supported providers: {providers}')

  provider_models = set(item.value for item in types.MODEL_MAP[provider])
  if provider_model not in provider_models:
    raise ValueError(
      f'Model {model} not supported for provider {provider}.\n'
      f'Supported models: {provider_models}')

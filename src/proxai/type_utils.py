import datetime
import proxai.types as types


def encode_datetime(dt: datetime.datetime) -> str:
  return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


def decode_datetime(dt_str: str) -> datetime.datetime:
  return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')


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

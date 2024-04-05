import os
from typing import Dict, Optional
import proxai.types as types
from proxai.connectors.model_connector import ModelConnector
from proxai.connectors.openai import OpenAIConnector
from proxai.connectors.claude import ClaudeConnector
from proxai.connectors.gemini import GeminiConnector
from proxai.connectors.cohere_api import CohereConnector
from proxai.connectors.databricks import DatabricksConnector
from proxai.connectors.mistral import MistralConnector
from proxai.connectors.hugging_face import HuggingFaceConnector
from proxai.logging.utils import LocalLoggingOptions

_RUN_TYPE: types.RunType = types.RunType.PRODUCTION
_REGISTERED_SIGNATURE: types.ModelSignature = None
_MODEL_CONNECTOR: ModelConnector = None
_LOCAL_LOGGING_OPTIONS: LocalLoggingOptions = LocalLoggingOptions()


def _set_run_type(run_type: types.RunType):
  global _RUN_TYPE
  _RUN_TYPE = run_type


def _get_model_connector_from_signature(
    signature: types.ModelSignature) -> ModelConnector:
  global _LOCAL_LOGGING_OPTIONS
  connector = None
  if signature.provider == types.Provider.OPENAI:
    connector =  OpenAIConnector
  elif signature.provider == types.Provider.CLAUDE:
    connector =  ClaudeConnector
  elif signature.provider == types.Provider.GEMINI:
    connector =  GeminiConnector
  elif signature.provider == types.Provider.COHERE:
    connector =  CohereConnector
  elif signature.provider == types.Provider.DATABRICKS:
    connector =  DatabricksConnector
  elif signature.provider == types.Provider.MISTRAL:
    connector =  MistralConnector
  elif signature.provider == types.Provider.HUGGING_FACE:
    connector =  HuggingFaceConnector
  else:
    raise ValueError("Provider not supported")

  if _LOCAL_LOGGING_OPTIONS.path:
    return connector(
        signature=signature,
        run_type=_RUN_TYPE,
        logging_options=_LOCAL_LOGGING_OPTIONS)

  return connector(
      signature=signature,
      run_type=_RUN_TYPE)

def _get_registered_model_connector() -> ModelConnector:
  global _REGISTERED_SIGNATURE
  global _MODEL_CONNECTOR
  if not _REGISTERED_SIGNATURE:
    _REGISTERED_SIGNATURE = types.ModelSignature(
        provider=types.Provider.OPENAI,
        model=types.OpenAIModel.GPT_3_5_TURBO)
    _MODEL_CONNECTOR = _get_model_connector_from_signature(
        _REGISTERED_SIGNATURE)
  return _MODEL_CONNECTOR


def local_logging_path(
      path: str,
      time: bool = True,
      prompt: bool = True,
      response: bool = True,
      error: bool = True):
  global _LOCAL_LOGGING_OPTIONS
  _LOCAL_LOGGING_OPTIONS.path = path
  _LOCAL_LOGGING_OPTIONS.time = time
  _LOCAL_LOGGING_OPTIONS.prompt = prompt
  _LOCAL_LOGGING_OPTIONS.response = response
  _LOCAL_LOGGING_OPTIONS.error = error


def register_model(provider: str, model: str):
  global _REGISTERED_SIGNATURE
  global _MODEL_CONNECTOR

  providers = set(item.value for item in types.Provider)
  if provider not in providers:
    raise ValueError(
      f'Provider not supported: {provider}. Supported providers: {providers}')

  if provider in [types.Provider.OPENAI,
                  types.Provider.CLAUDE,
                  types.Provider.GEMINI,
                  types.Provider.COHERE,
                  types.Provider.DATABRICKS,
                  types.Provider.MISTRAL,
                  types.Provider.HUGGING_FACE]:
    models = set(item.value for item in types.MODEL_MAP[provider])
    if model not in models:
      raise ValueError(
        f'Model {model} not supported for provider {provider}.\n'
        f'Supported models: {models}')
    _REGISTERED_SIGNATURE = types.ModelSignature(
        provider=types.Provider(provider),
        model=types.MODEL_MAP[provider](model))
  else:
    raise ValueError(
      'Provider not supported yet. We are looking for contributors!\n'
      f'Provider: {provider}')

  _MODEL_CONNECTOR = _get_model_connector_from_signature(
      _REGISTERED_SIGNATURE)


def generate_text(
    prompt: str,
    max_tokens: int = 100) -> str:
  model_connector = _get_registered_model_connector()
  return model_connector.generate_text(prompt, max_tokens)

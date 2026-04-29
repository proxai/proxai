"""Centralized provider API key management."""
from __future__ import annotations

import dataclasses
import os

import proxai.connections.proxdash as proxdash
import proxai.connectors.model_configs as model_configs
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_API_KEY_MANAGER_STATE_PROPERTY = '_api_key_manager_state'


@dataclasses.dataclass
class ApiKeyManagerParams:
  """Initialization parameters for ApiKeyManager."""

  proxdash_connection: proxdash.ProxDashConnection | None = None


class ApiKeyManager(state_controller.StateControlled):
  """Manages provider API keys from ProxDash and environment variables."""

  _api_key_manager_state: types.ApiKeyManagerState

  def __init__(
      self, init_from_params: ApiKeyManagerParams | None = None,
      init_from_state: types.ApiKeyManagerState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.proxdash_connection = init_from_params.proxdash_connection
      self.proxdash_provider_api_keys = (
          self.proxdash_connection.get_provider_api_keys()
          if self.proxdash_connection else {}
      )
      self.providers_with_key = {}
      self.load_provider_keys()

  def get_internal_state_property_name(self) -> str:
    return _API_KEY_MANAGER_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
    return types.ApiKeyManagerState

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(self, value: proxdash.ProxDashConnection):
    self.set_state_controlled_property_value('proxdash_connection', value)

  def proxdash_connection_deserializer(
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def proxdash_provider_api_keys(self) -> types.ProviderTokenValueMap:
    return self.get_property_value('proxdash_provider_api_keys')

  @proxdash_provider_api_keys.setter
  def proxdash_provider_api_keys(
      self, value: types.ProviderTokenValueMap | None
  ):
    self.set_property_value('proxdash_provider_api_keys', value)

  @property
  def providers_with_key(
      self
  ) -> dict[types.ProviderNameType, types.ProviderTokenValueMap]:
    return self.get_property_value('providers_with_key')

  @providers_with_key.setter
  def providers_with_key(
      self, value: dict[types.ProviderNameType, types.ProviderTokenValueMap]
  ):
    self.set_property_value('providers_with_key', value)

  def load_provider_keys(self):
    """Reload provider keys from ProxDash and environment variables.

    ProxDash keys take priority over environment variables. Called
    repeatedly by AvailableModels methods to pick up runtime changes.
    """
    self.providers_with_key = {}
    for provider, provider_key_names in model_configs.PROVIDER_KEY_MAP.items():
      for key_name in provider_key_names:
        if key_name in self.proxdash_provider_api_keys:
          if provider not in self.providers_with_key:
            self.providers_with_key[provider] = {}
          self.providers_with_key[provider][
              key_name] = self.proxdash_provider_api_keys[key_name]
        elif key_name in os.environ:
          if provider not in self.providers_with_key:
            self.providers_with_key[provider] = {}
          self.providers_with_key[provider][key_name] = os.environ[key_name]

  def get_provider_keys(
      self, provider: types.ProviderNameType
  ) -> types.ProviderTokenValueMap:
    """Get the token map for a specific provider, or empty dict."""
    return self.providers_with_key.get(provider, {})

  def has_provider_key(self, provider: types.ProviderNameType) -> bool:
    """Check if API keys are available for a provider."""
    return provider in self.providers_with_key

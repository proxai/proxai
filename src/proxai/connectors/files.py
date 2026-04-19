"""File management for provider File APIs."""

import dataclasses

import proxai.connections.proxdash as proxdash
import proxai.state_controllers.state_controller as state_controller
import proxai.types as types

_FILES_MANAGER_STATE_PROPERTY = '_files_manager_state'


@dataclasses.dataclass
class FilesManagerParams:
  """Initialization parameters for FilesManager."""

  run_type: types.RunType | None = None
  logging_options: types.LoggingOptions | None = None
  proxdash_connection: proxdash.ProxDashConnection | None = None
  provider_call_options: types.ProviderCallOptions | None = None


class FilesManager(state_controller.StateControlled):
  """Manages file uploads and references across provider File APIs."""

  _files_manager_state: types.FilesManagerState

  def __init__(
      self, init_from_params: FilesManagerParams | None = None,
      init_from_state: types.FilesManagerState | None = None
  ):
    super().__init__(
        init_from_params=init_from_params, init_from_state=init_from_state
    )

    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.run_type = init_from_params.run_type
      self.logging_options = init_from_params.logging_options
      self.proxdash_connection = init_from_params.proxdash_connection
      self.provider_call_options = init_from_params.provider_call_options

  def get_internal_state_property_name(self) -> str:
    return _FILES_MANAGER_STATE_PROPERTY

  def get_internal_state_type(self) -> type:
    return types.FilesManagerState

  @property
  def run_type(self) -> types.RunType:
    return self.get_property_value('run_type')

  @run_type.setter
  def run_type(self, run_type: types.RunType):
    self.set_property_value('run_type', run_type)

  @property
  def logging_options(self) -> types.LoggingOptions:
    return self.get_property_value('logging_options')

  @logging_options.setter
  def logging_options(self, logging_options: types.LoggingOptions):
    self.set_property_value('logging_options', logging_options)

  @property
  def proxdash_connection(self) -> proxdash.ProxDashConnection:
    return self.get_state_controlled_property_value('proxdash_connection')

  @proxdash_connection.setter
  def proxdash_connection(
      self, proxdash_connection: proxdash.ProxDashConnection
  ):
    self.set_state_controlled_property_value(
        'proxdash_connection', proxdash_connection
    )

  def proxdash_connection_deserializer(
      self, state_value: types.ProxDashConnectionState
  ) -> proxdash.ProxDashConnection:
    return proxdash.ProxDashConnection(init_from_state=state_value)

  @property
  def provider_call_options(self) -> types.ProviderCallOptions:
    return self.get_property_value('provider_call_options')

  @provider_call_options.setter
  def provider_call_options(
      self, provider_call_options: types.ProviderCallOptions
  ):
    self.set_property_value('provider_call_options', provider_call_options)

  def upload(self):
    pass

  def download(self):
    pass

  def list(self):
    pass

  def remove(self):
    pass

"""Base class for state controllers that manage the internal state of a class.

Example:
class UserStateController(StateController):
  @classmethod
  def get_internal_state_property_name(cls):
    return '_user_state'

class User:
  def __init__(self):
    self._user_state = {}  # Internal state storage
    self._name = None

  @property
  @UserStateController.getter
  def name(self):
    return self._name

  @name.setter
  @UserStateController.setter
  def name(self, value):
    self._name = value

# Usage
user = User()
user.name = "Alice"  # Sets name and automatically updates internal state
print(user._user_state)  # {'name': 'Alice'}
"""
from typing import Any, Optional
from abc import ABC, abstractmethod
import dataclasses
import proxai.types as types


class BaseStateControlled(ABC):
  def __init__(self, **kwargs):
    pass


class StateControlled(BaseStateControlled):
  def __init__(
      self,
      init_from_params=None,
      init_from_state=None):
    if init_from_params and init_from_state:
      raise ValueError(
          'init_from_params and init_from_state cannot be set at the same time.')

    if init_from_state is not None:
      self._validate_init_from_state_values(init_from_state)

    # Initialize the internal state structure.
    self.init_state_with_default_values()

  def _validate_init_from_state_values(self, init_from_state: Optional[Any]):
    if type(init_from_state) != self.get_internal_state_type():
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(init_from_state)}')

  @abstractmethod
  def get_internal_state_property_name(self):
    raise NotImplementedError('Subclasses must implement this method')

  @abstractmethod
  def get_internal_state_type(self):
    raise NotImplementedError('Subclasses must implement this method')

  @staticmethod
  def get_property_internal_name(field: str) -> str:
    return f'_{field}'

  @staticmethod
  def get_state_controlled_deserializer_name(field: str) -> str:
    return f'{field}_deserializer'

  def get_property_internal_value(self, property_name: str) -> Any:
    """Direct internal value getter."""
    return getattr(self, self.get_property_internal_name(property_name))

  def get_property_internal_state_value(self, property_name: str) -> Any:
    """Direct internal state value getter."""
    return getattr(
        getattr(self, self.get_internal_state_property_name()),
        property_name,
        None)

  def set_property_internal_value(
      self,
      property_name: str,
      value: Any):
    """Direct internal value setter."""
    setattr(self, self.get_property_internal_name(property_name), value)

  def set_property_internal_state_value(
      self,
      property_name: str,
      value: Any):
    """Sets the property value directly in the internal state."""

    setattr(
      getattr(self, self.get_internal_state_property_name()),
      property_name,
      value)

  def get_property_value(self, property_name: str) -> Any:
    result = self.get_property_internal_value(property_name)
    self.set_property_internal_state_value(property_name, result)
    return result

  def set_property_value(self, property_name: str, value: Any):
    self.set_property_internal_value(property_name, value)
    # Call actual getter to get the updated state value:
    updated_value = getattr(self, property_name)
    self.set_property_internal_state_value(property_name, updated_value)

  def get_state_controlled_property_value(self, property_name: str) -> Any:
    result = self.get_property_internal_value(property_name)

    if result is None:
      self.set_property_internal_state_value(property_name, None)
      return None

    if not isinstance(result, BaseStateControlled):
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(result)}')

    self.set_property_internal_state_value(property_name, result.get_state())
    return result

  def set_state_controlled_property_value(
      self,
      property_name: str,
      value: Any):
    if value is None:
      self.set_property_internal_value(property_name, None)
    elif isinstance(value, BaseStateControlled):
      self.set_property_internal_value(property_name, value)
    elif isinstance(value, types.StateContainer):
      value = getattr(
          self,
          self.get_state_controlled_deserializer_name(property_name))(value)
      self.set_property_internal_value(property_name, value)
    else:
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(value)}')

    # Call actual getter to get the updated state value:
    updated_value = getattr(self, property_name)

    if updated_value is None:
      self.set_property_internal_state_value(property_name, None)
      return

    if not isinstance(updated_value, BaseStateControlled):
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(updated_value)}')

    self.set_property_internal_state_value(
        property_name, updated_value.get_state())

  def set_property_value_without_triggering_getters(
      self,
      property_name: str,
      value: Any):
    self.set_property_internal_value(property_name, value)
    if isinstance(value, BaseStateControlled):
      value = value.get_state()
    self.set_property_internal_state_value(property_name, value)

  def init_state_with_default_values(self):
    setattr(
      self,
      self.get_internal_state_property_name(),
      self.get_internal_state_type()())

    for field in dataclasses.fields(self.get_internal_state_type()):
      self.set_property_value_without_triggering_getters(
          field.name, field.default)

    return self.get_state()

  def get_state(self) -> Any:
    result = self.get_internal_state_type()()
    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(self, field.name, None)
      if isinstance(value, BaseStateControlled):
        value = value.get_state()
      if value is not None:
        setattr(result, field.name, value)
    return result

  def get_internal_state(self) -> Any:
    return getattr(self, self.get_internal_state_property_name())

  def load_state(self, state: Any):
    if type(state) != self.get_internal_state_type():
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(state)}')

    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(state, field.name, None)
      if value is None:
        continue
      if isinstance(value, types.StateContainer):
        value = getattr(
            self,
            self.get_state_controlled_deserializer_name(field.name))(value)
      self.set_property_value_without_triggering_getters(field.name, value)

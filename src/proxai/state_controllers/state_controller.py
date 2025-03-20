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

Note: Current implementation makes deep copies if the value is not a primitive
type. Further optimizations can be made in the future for specific use cases.
"""
import copy
from functools import wraps
from typing import List, Callable, Any
from abc import ABC, abstractmethod
import dataclasses


class StateController(ABC):
  def __init__(self, **kwargs):
    available_properties = set([
        field.name
        for field in dataclasses.fields(self.get_internal_state_type())
    ])
    for raw_property_name, property_value in kwargs.items():
      property_name, property_getter_name = None, None
      if raw_property_name.startswith('get_'):
        property_name = self.get_property_name_from_func_getter_name(
            raw_property_name)
        property_getter_name = raw_property_name
      else:
        property_name = raw_property_name
        property_getter_name = self.get_property_func_getter_name(
            raw_property_name)

      if property_value is None:
        continue

      if property_name not in available_properties:
        raise ValueError(
            f'Invalid property name or property getter name:\n'
            f'Property name: {property_name}\n'
            f'Property getter name: {property_getter_name}\n'
            f'Available properties: {available_properties}')

      if (property_name in kwargs and
          kwargs[property_name] is not None and
          property_getter_name in kwargs and
          kwargs[property_getter_name] is not None):
        raise ValueError(
            f'Only one of {property_name} or {property_getter_name} should be set '
            'while initializing the StateControlled object.')

  @abstractmethod
  def get_internal_state_property_name(cls):
    raise NotImplementedError('Subclasses must implement this method')

  @abstractmethod
  def get_internal_state_type(cls):
    raise NotImplementedError('Subclasses must implement this method')

  @abstractmethod
  def handle_changes(
      self,
      old_state: dataclasses.dataclass,
      current_state: dataclasses.dataclass):
    raise NotImplementedError(
        'Subclasses must implement this method before using it.')

  @staticmethod
  def get_property_internal_name(field: str) -> str:
    return f'_{field}'

  @staticmethod
  def get_property_func_internal_getter_name(field: str) -> str:
    return f'_get_{field}'

  @staticmethod
  def get_property_func_getter_name(field: str) -> str:
    return f'get_{field}'

  @staticmethod
  def get_property_name_from_func_getter_name(func_getter_name: str) -> str:
    if not func_getter_name.startswith('get_'):
      raise ValueError(
          f'Invalid function getter name: {func_getter_name}. '
          'It should start with "get_".')
    return func_getter_name[4:]

  def get_property_internal_value(self, property_name: str) -> Any:
    """Direct internal value getter."""
    return getattr(self, self.get_property_internal_name(property_name))

  def get_property_internal_value_or_from_getter(
      self, property_name: str) -> Any:
    """Getter for the properties that support both direct and getter values.

    Returns the property value if it is set directly, otherwise returns the
    value from the getter function. If the property is not set directly or by
    the getter function, returns None.
    """
    literal_value = getattr(
        self,
        self.get_property_internal_name(property_name),
        None)
    if literal_value is not None:
      return literal_value

    func_value = getattr(
        self,
        self.get_property_func_internal_getter_name(property_name),
        None)
    if func_value is not None:
      return func_value()

    return None

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
    # Note: This is not efficient but safely copies the value to the internal
    # state. In the future, we can optimize this further for specific use cases.
    if value is None:
      copied_value = None
    elif isinstance(value, (str, int, float, bool, type(None))):
      copied_value = value
    else:
      copied_value = copy.deepcopy(value)
    setattr(
      getattr(self, self.get_internal_state_property_name()),
      property_name,
      copied_value)

  def get_property_value(self, property_name: str) -> Any:
    result = self.get_property_internal_value_or_from_getter(property_name)
    self.set_property_internal_state_value(property_name, result)
    return result

  def set_property_value(self, property_name: str, value: Any):
    self.set_property_internal_value(property_name, value)
    # Call actual getter to get the updated state value:
    updated_value = getattr(self, property_name)
    self.set_property_internal_state_value(property_name, updated_value)

  def set_property_value_without_triggering_getters(
      self,
      property_name: str,
      value: Any):
    self.set_property_internal_value(property_name, value)
    self.set_property_internal_state_value(property_name, value)

  def init_state(self):
    setattr(
      self,
      self.get_internal_state_property_name(),
      self.get_internal_state_type()())

    for field in dataclasses.fields(self.get_internal_state_type()):
      self.set_property_value_without_triggering_getters(
          field.name, field.default)
      if hasattr(self, self.get_property_func_internal_getter_name(field.name)):
        setattr(
          self, self.get_property_func_internal_getter_name(field.name), None)

    return self.get_state()

  def get_state(self) -> Any:
    result = self.get_internal_state_type()()
    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(self, field.name, None)
      if value is not None:
        setattr(result, field.name, value)
    return result

  def get_internal_state(self) -> Any:
    return copy.deepcopy(
        getattr(self, self.get_internal_state_property_name()))

  def get_external_state_changes(self) -> Any:
    internal_state = self.get_internal_state()
    changes = self.get_internal_state_type()()

    for field in dataclasses.fields(self.get_internal_state_type()):
      if getattr(internal_state, field.name, None) != getattr(self, field.name):
        setattr(changes, field.name, getattr(self, field.name))
    return changes

  def load_state(self, state: Any):
    if type(state) != self.get_internal_state_type():
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(state)}')

    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(state, field.name, None)
      if value is not None:
        self.set_property_value_without_triggering_getters(field.name, value)

  def apply_state_changes(self, changes: Any):
    if changes is None:
      changes = self.get_internal_state_type()()

    if type(changes) != self.get_internal_state_type():
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(changes)}')

    old_state = self.get_internal_state()
    self.load_state(changes)
    self.handle_changes(old_state, self.get_internal_state())

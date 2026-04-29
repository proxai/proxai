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

from __future__ import annotations

import copy
import dataclasses
import inspect
import types as _builtin_types
import typing
from abc import abstractmethod
from typing import Any

import proxai.types as types

def _unwrap_optional(type_hint: Any) -> Any | None:
  """Extract the non-None type from Optional/Union annotations."""
  origin = typing.get_origin(type_hint)
  # On Python 3.10-3.13, `X | Y` syntax creates `types.UnionType`, distinct
  # from `typing.Union` (returned by `Union[X, Y]`). Python 3.14 unified
  # them, so `is typing.Union` alone is sufficient there. Accept both.
  if origin is typing.Union or origin is _builtin_types.UnionType:
    args = [a for a in typing.get_args(type_hint) if a is not type(None)]
    return args[0] if len(args) == 1 else None
  return type_hint


class BaseStateControlled:
  """Base class for objects with serializable state."""

  def __init__(self, **kwargs):
    pass


class StateControlled(BaseStateControlled):
  """Mixin providing automatic state serialization for class properties."""

  def __init__(self, init_from_params=None, init_from_state=None):
    if init_from_params and init_from_state:
      raise ValueError(
          'init_from_params and init_from_state cannot be set '
          'at the same time.'
      )

    if init_from_state is not None:
      self._validate_init_from_state_values(init_from_state)

    # Initialize the internal state structure.
    self.init_state_with_default_values()
    self._validate_state_contract()

  def _validate_init_from_state_values(self, init_from_state: Any | None):
    if not isinstance(init_from_state, self.get_internal_state_type()):
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(init_from_state)}'
      )

  def _validate_state_contract(self):
    """Validate that state fields, properties, and deserializers are in sync.

    Enforces two invariants:
      1. Every state field must have a @property on the class so that
         get_state() can read it and load_state() can restore it.
      2. Every StateContainer-typed state field must have a
         {field_name}_deserializer method so load_state() can reconstruct
         the object from serialized state.

    Runs on every instantiation. The cost is negligible (iterating a
    handful of dataclass fields with getattr checks). If batch or
    multiprocessing workloads create many instances of the same class,
    add a per-class cache flag to skip repeated validation:
        if '_state_contract_validated' in cls.__dict__:
          return
        ...
        cls._state_contract_validated = True
    """
    cls = type(self)
    state_type = self.get_internal_state_type()
    try:
      hints = typing.get_type_hints(state_type)
    except Exception:
      return

    errors = []
    for field in dataclasses.fields(state_type):
      prop = inspect.getattr_static(cls, field.name, None)
      if not isinstance(prop, property):
        errors.append(
            f"State field '{field.name}' (in {state_type.__name__}) "
            f"has no @property on {cls.__name__}. Without a property, "
            f"get_state() cannot read this field and load_state() "
            f"cannot restore it. Add a @property getter/setter pair "
            f"using get_property_value/set_property_value."
        )

      field_type = _unwrap_optional(hints.get(field.name))
      if (
          isinstance(field_type, type) and
          issubclass(field_type, types.StateContainer)
      ):
        deserializer_name = self.get_state_controlled_deserializer_name(
            field.name
        )
        if not callable(getattr(cls, deserializer_name, None)):
          errors.append(
              f"State field '{field.name}' (type: "
              f"{field_type.__name__}) is a StateContainer but "
              f"{cls.__name__} has no '{deserializer_name}' method. "
              f"load_state() will crash when reconstructing from "
              f"serialized state."
          )

    if errors:
      raise TypeError(
          f"StateControlled contract violations in {cls.__name__}:\n"
          + "\n".join(f"  - {e}" for e in errors)
      )

  @abstractmethod
  def get_internal_state_property_name(self):
    """Return the name of the internal state property."""
    raise NotImplementedError('Subclasses must implement this method')

  @abstractmethod
  def get_internal_state_type(self):
    """Return the dataclass type used for state storage."""
    raise NotImplementedError('Subclasses must implement this method')

  @staticmethod
  def get_property_internal_name(field: str) -> str:
    """Return the internal storage name for a field."""
    return f'_{field}'

  @staticmethod
  def get_state_controlled_deserializer_name(field: str) -> str:
    """Return the deserializer method name for a field."""
    return f'{field}_deserializer'

  def get_property_internal_value(self, property_name: str) -> Any:
    """Direct internal value getter."""
    return getattr(self, self.get_property_internal_name(property_name))

  def get_property_internal_state_value(self, property_name: str) -> Any:
    """Direct internal state value getter."""
    return getattr(
        getattr(self, self.get_internal_state_property_name()), property_name,
        None
    )

  def set_property_internal_value(self, property_name: str, value: Any):
    """Direct internal value setter."""
    setattr(self, self.get_property_internal_name(property_name), value)

  def set_property_internal_state_value(self, property_name: str, value: Any):
    """Sets the property value directly in the internal state."""
    setattr(
        getattr(self, self.get_internal_state_property_name()), property_name,
        value
    )

  def get_property_value(self, property_name: str) -> Any:
    """Get a property value and sync it to internal state."""
    result = self.get_property_internal_value(property_name)
    self.set_property_internal_state_value(property_name, result)
    return result

  def set_property_value(self, property_name: str, value: Any):
    """Set a property value and sync it to internal state."""
    self.set_property_internal_value(property_name, value)
    # Call actual getter to get the updated state value:
    updated_value = getattr(self, property_name)
    self.set_property_internal_state_value(property_name, updated_value)

  def get_state_controlled_property_value(self, property_name: str) -> Any:
    """Get a nested StateControlled property and sync its state."""
    result = self.get_property_internal_value(property_name)

    if result is None:
      self.set_property_internal_state_value(property_name, None)
      return None

    if not isinstance(result, BaseStateControlled):
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(result)}'
      )

    self.set_property_internal_state_value(property_name, result.get_state())
    return result

  def set_state_controlled_property_value(self, property_name: str, value: Any):
    """Set a nested StateControlled property from value or state."""
    if value is None:
      self.set_property_internal_value(property_name, None)
    elif isinstance(value, BaseStateControlled):
      self.set_property_internal_value(property_name, value)
    elif isinstance(value, types.StateContainer):
      value = getattr(
          self, self.get_state_controlled_deserializer_name(property_name)
      )(value)
      self.set_property_internal_value(property_name, value)
    else:
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(value)}'
      )

    # Call actual getter to get the updated state value:
    updated_value = getattr(self, property_name)

    if updated_value is None:
      self.set_property_internal_state_value(property_name, None)
      return

    if not isinstance(updated_value, BaseStateControlled):
      raise ValueError(
          f'Invalid property value. Expected a StateControlled object. '
          f'Got: {type(updated_value)}'
      )

    self.set_property_internal_state_value(
        property_name, updated_value.get_state()
    )

  def set_property_value_without_triggering_getters(
      self, property_name: str, value: Any
  ):
    """Set a property value directly without invoking computed getters."""
    self.set_property_internal_value(property_name, value)
    if isinstance(value, BaseStateControlled):
      value = value.get_state()
    self.set_property_internal_state_value(property_name, value)

  def init_state_with_default_values(self):
    """Initialize the internal state with default field values."""
    setattr(
        self, self.get_internal_state_property_name(),
        self.get_internal_state_type()()
    )

    for field in dataclasses.fields(self.get_internal_state_type()):
      self.set_property_value_without_triggering_getters(
          field.name, field.default
      )

    return self.get_state()

  def get_state(self) -> Any:
    """Return a snapshot of the current state as a dataclass."""
    result = self.get_internal_state_type()()
    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(self, field.name, None)
      if isinstance(value, BaseStateControlled):
        value = value.get_state()
      if value is not None:
        setattr(result, field.name, value)
    return result

  def clone_state(self) -> Any:
    """Return a deep-copied snapshot of the current state.

    Unlike get_state(), the returned dataclass shares no mutable references
    with the original object, so callers can mutate it freely without
    affecting this instance.
    """
    return copy.deepcopy(self.get_state())

  def get_internal_state(self) -> Any:
    """Return the raw internal state object."""
    return getattr(self, self.get_internal_state_property_name())

  def load_state(self, state: Any):
    """Restore object state from a previously saved state dataclass."""
    if not isinstance(state, self.get_internal_state_type()):
      raise ValueError(
          f'Invalid state type.\nExpected: {self.get_internal_state_type()}\n'
          f'Actual: {type(state)}'
      )

    for field in dataclasses.fields(self.get_internal_state_type()):
      value = getattr(state, field.name, None)
      if value is None:
        continue
      if isinstance(value, types.StateContainer):
        value = getattr(
            self, self.get_state_controlled_deserializer_name(field.name)
        )(value)
      self.set_property_value_without_triggering_getters(field.name, value)

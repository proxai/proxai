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
from typing import List, Callable


class StateController:
  @classmethod
  def get_internal_state_property_name(cls):
    raise NotImplementedError('Subclasses must implement this method')

  @classmethod
  def getter(cls, func: Callable) -> Callable:
    @wraps(func)
    def wrapper(instance):
      result = func(instance)
      internal_state = getattr(
          instance, cls.get_internal_state_property_name())

      # Efficiently handle value copying based on type
      if result is None:
        copied_value = None
      elif isinstance(result, (str, int, float, bool, type(None))):
        copied_value = result
      else:
        copied_value = copy.deepcopy(result)

      setattr(internal_state, func.__name__, copied_value)
      return result
    return wrapper

  @classmethod
  def setter(cls, func: Callable) -> Callable:
    @wraps(func)
    def wrapper(instance, value):
      result = func(instance, value)
      internal_state_value = getattr(instance, func.__name__)
      internal_state = getattr(
          instance, cls.get_internal_state_property_name())

      # Efficiently handle value copying based on type
      if internal_state_value is None:
        copied_value = None
      elif isinstance(
          internal_state_value, (str, int, float, bool, type(None))):
        copied_value = internal_state_value
      else:
        copied_value = copy.deepcopy(internal_state_value)

      setattr(internal_state, func.__name__, copied_value)
      return result
    return wrapper

  @classmethod
  def requires_dependencies(
      cls,
      required_props: List[str]) -> Callable:
    def decorator(func):
      @wraps(func)
      def wrapper(instance, *args):
        if len(args) == 0:
          return func(instance)
        if len(args) > 0 and args[0] is None:
          return func(instance, *args)
        for prop in required_props:
          if not hasattr(instance, prop) or getattr(instance, prop) is None:
            raise ValueError(
                f'Property {prop} must be set before calling {func.__name__}.\n'
                'Current internal state: '
                f'{getattr(instance, cls.get_internal_state_property_name())}')
        return func(instance, *args)
      return wrapper
    return decorator

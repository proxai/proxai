import pytest
import proxai.state_controllers.state_controller as state_controller
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Tuple


@dataclass
class ExampleState:
  property_1: Optional[Any] = None
  property_2: Optional[Any] = None
  property_3: Optional[Any] = None


class ExampleStateControlledClass(state_controller.StateController):
  _property_1: Any
  _property_2: Any
  _get_property_2: Callable[[], Any]
  _property_3: Any
  _get_property_3: Callable[[], Any]
  _state: ExampleState
  property_3_getter_called: bool
  property_3_setter_called: bool
  change_list: List[Tuple[str, Any, Any]]

  def __init__(
      self,
      property_1: Optional[str] = None,
      property_2: Optional[str] = None,
      get_property_2: Optional[Callable[[], str]] = None,
      property_3: Optional[str] = None,
      get_property_3: Optional[Callable[[], str]] = None,
      non_available_property: Optional[str] = None):

    super().__init__(
        property_1=property_1,
        property_2=property_2,
        get_property_2=get_property_2,
        property_3=property_3,
        get_property_3=get_property_3,
        non_available_property=non_available_property)

    init_state = self.init_state()

    self.property_3_getter_called = False
    self.property_3_setter_called = False
    self.change_list = []

    self._property_1 = property_1
    self._property_2 = property_2
    self._get_property_2 = get_property_2
    self._property_3 = property_3
    self._get_property_3 = get_property_3

    self.handle_changes(init_state, self.get_state())

  def get_internal_state_property_name(cls):
    return '_state'

  def get_internal_state_type(cls):
    return ExampleState

  def handle_changes(
      self,
      old_state: ExampleState,
      current_state: ExampleState):
    if old_state.property_1 != current_state.property_1:
      self.change_list.append(
          ('property_1', old_state.property_1, current_state.property_1))
    if old_state.property_2 != current_state.property_2:
      self.change_list.append(
          ('property_2', old_state.property_2, current_state.property_2))
    if old_state.property_3 != current_state.property_3:
      self.change_list.append(
          ('property_3', old_state.property_3, current_state.property_3))

  @property
  def property_1(self):
    return self.get_property_value('property_1')

  @property_1.setter
  def property_1(self, value):
    self.set_property_value('property_1', value)

  @property
  def property_2(self):
    return self.get_property_value('property_2')

  @property_2.setter
  def property_2(self, value):
    self.set_property_value('property_2', value)

  @property
  def property_3(self):
    self.property_3_getter_called = True
    return self.get_property_value('property_3')

  @property_3.setter
  def property_3(self, value):
    self.property_3_setter_called = True
    self.set_property_value('property_3', value)


class TestStateControlled:
  def test_get_property_func_conversions(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.get_property_internal_name(
        'property_1') == '_property_1'
    assert example_obj.get_property_func_internal_getter_name(
        'property_1') == '_get_property_1'
    assert example_obj.get_property_func_getter_name(
        'property_1') == 'get_property_1'
    assert example_obj.get_property_name_from_func_getter_name(
        'get_property_1') == 'property_1'

  def test_get_property_internal_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj._get_property_3 = lambda: 'test2'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    assert example_obj.get_property_internal_value('property_3') == 'test1'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_get_property_internal_value_or_from_getter(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj._get_property_3 = lambda: 'test2'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    assert example_obj.get_property_internal_value_or_from_getter(
        'property_3') == 'test1'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

    example_obj._property_3 = None
    assert example_obj.get_property_internal_value_or_from_getter(
        'property_3') == 'test2'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

    example_obj._get_property_3 = None
    assert example_obj.get_property_internal_value_or_from_getter(
        'property_3') is None
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_get_property_internal_state_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj._state.property_3 = 'test'
    assert example_obj.get_property_internal_value('property_3') is None
    assert example_obj.get_property_internal_state_value('property_3') == 'test'


  def test_set_property_internal_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    example_obj.set_property_internal_value('property_3', 'test')
    assert example_obj._property_3 == 'test'
    assert example_obj._state.property_3 is None
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_set_property_internal_state_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    example_obj.set_property_internal_state_value('property_3', 'test')
    assert example_obj._property_3 is None
    assert example_obj._state.property_3 == 'test'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_get_property_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    assert example_obj.get_property_value('property_3') == 'test1'
    assert example_obj._property_3 == 'test1'
    assert example_obj._state.property_3 == 'test1'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_set_property_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    example_obj.set_property_value('property_3', 'test2')
    assert example_obj._property_3 == 'test2'
    assert example_obj._state.property_3 == 'test2'
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

  def test_set_property_value_without_triggering_getters(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    example_obj.set_property_value_without_triggering_getters(
        'property_3', 'test2')
    assert example_obj._property_3 == 'test2'
    assert example_obj._state.property_3 == 'test2'
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_conflicting_init_args(self):
    with pytest.raises(
        ValueError,
        match='Only one of property_2 or get_property_2 should be set '
        'while initializing the StateControlled object.'):
      ExampleStateControlledClass(
          property_1='test',
          property_2='test_2',
          get_property_2=lambda: 'test_3')

  def test_non_available_field_name(self):
    with pytest.raises(
        ValueError,
        match='Invalid property name or property getter name:\n'
        'Property name: non_available_property\n'
        'Property getter name: get_non_available_property\n'
        'Available properties:'):
      ExampleStateControlledClass(non_available_property='test')

  def test_init_state(self):
    @dataclass
    class ExampleState2(ExampleState):
      property_1: Optional[bool] = True
      property_2: Optional[bool] = False
      property_3: Optional[bool] = None

    class ExampleStateControlledClass2(ExampleStateControlledClass):
      _state: ExampleState2

      def get_internal_state_type(self):
        return ExampleState2

    example_obj = ExampleStateControlledClass2()
    assert example_obj.get_internal_state() == ExampleState2(
        property_1=None,
        property_2=None,
        property_3=None)
    example_obj.init_state()
    assert example_obj.get_internal_state() == ExampleState2(
        property_1=True,
        property_2=False,
        property_3=None)
    assert example_obj._get_property_2 is None
    assert example_obj._get_property_3 is None

  def test_get_state(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    example_obj.property_3 = 'test'
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3='test')
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == True

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    example_obj.set_property_value_without_triggering_getters(
        'property_3', 'test_2')
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3='test_2')
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

  def test_get_internal_state(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    example_obj.property_3 = 'test'
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3='test')
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == True

    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False
    example_obj.set_property_value_without_triggering_getters(
        'property_3', 'test_2')
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3='test_2')
    assert example_obj.property_3_getter_called == False
    assert example_obj.property_3_setter_called == False

  def test_get_external_state_changes_primitive_types(self):
    dynamic_property_2_value = 'value_1'
    dynamic_property_3_value = 'value_1'
    example_obj = ExampleStateControlledClass(
        get_property_2=lambda: dynamic_property_2_value,
        get_property_3=lambda: dynamic_property_3_value)
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2='value_1',
        property_3='value_1')

    dynamic_property_2_value = 'value_2'
    dynamic_property_3_value = 'value_2'
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2='value_1',
        property_3='value_1')
    # This should return True as the values are updated.
    assert example_obj.get_external_state_changes() == ExampleState(
        property_2='value_2',
        property_3='value_2')
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2='value_2',
        property_3='value_2')
    # Second call should return empty changes as the values are already updated.
    assert example_obj.get_external_state_changes() == ExampleState()

  def test_get_external_state_changes_non_primitive_types(self):
    dynamic_property_2_value = {'property_2': 'value_1'}
    dynamic_property_3_value = {'property_3': 'value_1'}
    example_obj = ExampleStateControlledClass(
        get_property_2=lambda: dynamic_property_2_value,
        get_property_3=lambda: dynamic_property_3_value)
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_1'},
        property_3={'property_3': 'value_1'})

    dynamic_property_2_value['property_2'] = 'value_2'
    dynamic_property_3_value['property_3'] = 'value_2'
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_1'},
        property_3={'property_3': 'value_1'})
    # This should return True as the values are updated.
    assert example_obj.get_external_state_changes() == ExampleState(
        property_2={'property_2': 'value_2'},
        property_3={'property_3': 'value_2'})
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_2'},
        property_3={'property_3': 'value_2'})
    # Second call should return empty changes as the values are already updated.
    assert example_obj.get_external_state_changes() == ExampleState()

  def test_load_state_incorrect_state_type(self):
    with pytest.raises(
        ValueError,
        match='Invalid state type.\nExpected: <class'):
      ExampleStateControlledClass().load_state('test')

  def test_load_state(self):
    example_obj = ExampleStateControlledClass(
        property_1='test',
        get_property_2=lambda: 'test_2',
        get_property_3=lambda: 'test_3')
    current_state = example_obj.get_state()
    assert current_state == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == False

    # Load state should not call the getters.
    example_obj_2 = ExampleStateControlledClass()

    example_obj_2.property_3_getter_called = False
    example_obj_2.property_3_setter_called = False
    example_obj_2.load_state(current_state)
    assert example_obj_2.get_internal_state() == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')
    assert example_obj_2.property_3_getter_called == False
    assert example_obj_2.property_3_setter_called == False

  def test_load_state_partial_values(self):
    example_obj = ExampleStateControlledClass(
        property_2='test_2')
    current_state = example_obj.get_state()
    assert current_state == ExampleState(
        property_1=None,
        property_2='test_2',
        property_3=None)

    example_obj_2 = ExampleStateControlledClass()
    example_obj_2.load_state(current_state)
    assert example_obj_2.get_internal_state() == ExampleState(
        property_1=None,
        property_2='test_2',
        property_3=None)

  def test_apply_state_changes_incorrect_state_type(self):
    with pytest.raises(
        ValueError,
        match='Invalid state type.\nExpected: <class'):
      ExampleStateControlledClass().apply_state_changes('test')

  def test_apply_state_changes(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)

    example_obj.apply_state_changes(ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_1',
        property_3='property_3_value_1'))
    assert example_obj.get_internal_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_1',
        property_3='property_3_value_1')
    assert example_obj.change_list == [
        ('property_1', None, 'property_1_value_1'),
        ('property_2', None, 'property_2_value_1'),
        ('property_3', None, 'property_3_value_1')]

    example_obj.apply_state_changes(ExampleState(
        property_2='property_2_value_2'))
    assert example_obj.get_internal_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_2',
        property_3='property_3_value_1')
    assert example_obj.change_list == [
        ('property_1', None, 'property_1_value_1'),
        ('property_2', None, 'property_2_value_1'),
        ('property_3', None, 'property_3_value_1'),
        ('property_2', 'property_2_value_1', 'property_2_value_2')]

  def test_apply_state_changes_getter_func(self):
    dynamic_property_2_value = {'property_2': 'value_1'}
    dynamic_property_3_value = {'property_3': 'value_1'}
    example_obj = ExampleStateControlledClass(
        get_property_2=lambda: dynamic_property_2_value,
        get_property_3=lambda: dynamic_property_3_value)
    assert example_obj.change_list == [
        ('property_2', None, {'property_2': 'value_1'}),
        ('property_3', None, {'property_3': 'value_1'})]

    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_1'},
        property_3={'property_3': 'value_1'})
    assert example_obj.change_list == [
        ('property_2', None, {'property_2': 'value_1'}),
        ('property_3', None, {'property_3': 'value_1'})]

    example_obj.apply_state_changes(ExampleState(
        property_2='value_2',
        property_3='value_2'))
    assert example_obj.change_list == [
        ('property_2', None, {'property_2': 'value_1'}),
        ('property_3', None, {'property_3': 'value_1'}),
        ('property_2', {'property_2': 'value_1'}, 'value_2'),
        ('property_3', {'property_3': 'value_1'}, 'value_2')]

    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2='value_2',
        property_3='value_2')
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2='value_2',
        property_3='value_2')
    assert example_obj.change_list == [
        ('property_2', None, {'property_2': 'value_1'}),
        ('property_3', None, {'property_3': 'value_1'}),
        ('property_2', {'property_2': 'value_1'}, 'value_2'),
        ('property_3', {'property_3': 'value_1'}, 'value_2')]

  def test_property_setter(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)

    # These setters should update the internal state.
    example_obj.property_1 = 'test'
    example_obj.property_2 = 'test_2'
    example_obj.property_3 = 'test_3'

    # Internal state should be updated.
    assert example_obj.get_internal_state() == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')

  def test_property_getter(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_1 = 'test'
    example_obj._property_2 = 'test_2'
    example_obj._property_3 = 'test_3'

    # Internal state not changed yet.
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)

    # This should update the internal state also.
    assert example_obj.get_state() == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')

    # Internal state should be updated after get_state is called.
    assert example_obj.get_internal_state() == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')

  def test_property_getter_with_function(self):
    dynamic_property_2_value = 'property_2_value_1'
    dynamic_property_3_value = 'property_3_value_1'
    example_obj = ExampleStateControlledClass()
    example_obj._property_1 = 'property_1_value_1'
    example_obj._get_property_2 = lambda: dynamic_property_2_value
    example_obj._get_property_3 = lambda: dynamic_property_3_value
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None)

    # This should update the internal state.
    assert example_obj.get_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_1',
        property_3='property_3_value_1')

    # Internal state should be updated after get_state is called.
    assert example_obj.get_internal_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_1',
        property_3='property_3_value_1')

    dynamic_property_2_value = 'property_2_value_2'

    # Internal state should not be updated yet.
    assert example_obj.get_internal_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_1',
        property_3='property_3_value_1')

    # This should update the internal state.
    assert example_obj.get_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_2',
        property_3='property_3_value_1')

    # Internal state should be updated after get_state is called.
    assert example_obj.get_internal_state() == ExampleState(
        property_1='property_1_value_1',
        property_2='property_2_value_2',
        property_3='property_3_value_1')

  def test_property_getter_with_function_non_primitive_types(self):
    dynamic_property_2_value = {'property_2': 'value_1'}
    dynamic_property_3_value = {'property_3': 'value_1'}
    example_obj = ExampleStateControlledClass(
        get_property_2=lambda: dynamic_property_2_value,
        get_property_3=lambda: dynamic_property_3_value)
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_1'},
        property_3={'property_3': 'value_1'})

    dynamic_property_2_value['property_2'] = 'value_2'
    dynamic_property_3_value['property_3'] = 'value_2'
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_1'},
        property_3={'property_3': 'value_1'})
    assert example_obj.get_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_2'},
        property_3={'property_3': 'value_2'})
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2={'property_2': 'value_2'},
        property_3={'property_3': 'value_2'})

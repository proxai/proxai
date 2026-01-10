import pytest
import proxai.state_controllers.state_controller as state_controller
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Tuple
import proxai.types as types


@dataclass
class ExampleSubState(types.StateContainer):
  sub_property_1: Optional[Any] = None


@dataclass
class ExampleSubStateParams:
  sub_property_1: Optional[Any] = None


class ExampleSubStateControlledClass(state_controller.StateControlled):
  def __init__(
      self,
      init_from_params: Optional[ExampleSubStateParams] = None,
      init_from_state: Optional[ExampleSubState] = None):
    super().__init__(
        init_from_params=init_from_params,
        init_from_state=init_from_state)

    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.sub_property_1 = init_from_params.sub_property_1

  def get_internal_state_property_name(cls):
    return '_state'

  def get_internal_state_type(cls):
    return ExampleSubState

  @property
  def sub_property_1(self):
    return self.get_property_value('sub_property_1')

  @sub_property_1.setter
  def sub_property_1(self, value):
    self.set_property_value('sub_property_1', value)


@dataclass
class ExampleState(types.StateContainer):
  property_1: Optional[Any] = None
  property_2: Optional[Any] = None
  property_3: Optional[Any] = None
  sub_state: Optional[ExampleSubState] = None

@dataclass
class ExampleStateParams:
  property_1: Optional[Any] = None
  property_2: Optional[Any] = None
  property_3: Optional[Any] = None
  sub_state: Optional[ExampleSubStateControlledClass] = None


class ExampleStateControlledClass(state_controller.StateControlled):
  def __init__(
      self,
      init_from_params: Optional[ExampleSubStateParams] = None,
      init_from_state: Optional[ExampleSubState] = None):
    super().__init__(
        init_from_params=init_from_params,
        init_from_state=init_from_state)
    self.property_3_getter_called = False
    self.property_3_setter_called = False
    self.sub_state_getter_called = False

    self.change_list = []
    if init_from_state:
      self.load_state(init_from_state)
    elif init_from_params:
      self.property_1 = init_from_params.property_1
      self.property_2 = init_from_params.property_2
      self.property_3 = init_from_params.property_3
      self.sub_state = init_from_params.sub_state

  def get_internal_state_property_name(cls):
    return '_state'

  def get_internal_state_type(cls):
    return ExampleState

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

  @property
  def sub_state(self):
    self.sub_state_getter_called = True
    return self.get_state_controlled_property_value('sub_state')

  @sub_state.setter
  def sub_state(self, value):
    self.set_state_controlled_property_value('sub_state', value)

  def sub_state_deserializer(
      self,
      state_value: ExampleSubState):
    return ExampleSubStateControlledClass(
        init_from_params=ExampleSubStateParams(
            sub_property_1=state_value.sub_property_1))


class TestStateControlled:
  def test_get_property_func_conversions(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.get_property_internal_name(
        'property_1') == '_property_1'
    assert example_obj.get_state_controlled_deserializer_name(
        'property_1') == 'property_1_deserializer'

  def test_get_property_internal_value(self):
    example_obj = ExampleStateControlledClass()
    example_obj._property_3 = 'test1'
    example_obj.property_3_getter_called = False
    example_obj.property_3_setter_called = False

    assert example_obj.get_property_internal_value('property_3') == 'test1'
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
        match='init_from_params and init_from_state cannot be set at the same time.'):
      ExampleStateControlledClass(
          init_from_params=ExampleStateParams(
              property_1='test',
              property_2='test_2',
              property_3='test_3'),
          init_from_state=ExampleState(
              property_1='test',
              property_2='test_2',
              property_3='test_3'))

  def test_auto_init_state_called(self):
    """Test that init_state() is automatically called by parent __init__."""
    @dataclass
    class MinimalState(types.StateContainer):
      value: Optional[str] = None

    class MinimalClass(state_controller.StateControlled):
      def __init__(self):
        # Don't call init_state() - rely on parent to call it
        super().__init__()

      def get_internal_state_property_name(self):
        return '_minimal_state'

      def get_internal_state_type(self):
        return MinimalState

      def handle_changes(self, old_state, current_state):
        pass

    # If parent didn't call init_state(), this would fail
    obj = MinimalClass()
    assert hasattr(obj, '_minimal_state')
    assert obj._minimal_state is not None
    assert isinstance(obj._minimal_state, MinimalState)

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
        property_1=True,
        property_2=False,
        property_3=None)

  def test_get_state(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.property_3_getter_called == False
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
    assert example_obj.property_3_getter_called == False
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

  def test_load_state_incorrect_state_type(self):
    with pytest.raises(
        ValueError,
        match='Invalid state type.\nExpected: <class'):
      ExampleStateControlledClass().load_state('test')

  def test_load_state(self):
    example_params = ExampleStateParams(
        property_1='test',
        property_2='test_2',
        property_3='test_3')
    example_obj = ExampleStateControlledClass(
        init_from_params=example_params)
    current_state = example_obj.get_state()
    assert current_state == ExampleState(
        property_1='test',
        property_2='test_2',
        property_3='test_3')
    assert example_obj.property_3_getter_called == True
    assert example_obj.property_3_setter_called == True

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
    example_params = ExampleStateParams(
        property_2='test_2')
    example_obj = ExampleStateControlledClass(
        init_from_params=example_params)
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


class TestSubState:
  def test_sub_state_literal(self):
    example_sub_state_params = ExampleSubStateParams(
        sub_property_1='sub_property_1_value_1')
    example_sub_state_obj = ExampleSubStateControlledClass(
        init_from_params=example_sub_state_params)
    example_params = ExampleStateParams(
        sub_state=example_sub_state_obj)
    example_obj = ExampleStateControlledClass(
        init_from_params=example_params)
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None,
        sub_state=ExampleSubState(
            sub_property_1='sub_property_1_value_1'))
    assert example_obj.sub_state.sub_property_1 == 'sub_property_1_value_1'

  def test_sub_state_setter(self):
    example_obj = ExampleStateControlledClass()
    assert example_obj.sub_state is None

    example_sub_state_params = ExampleSubStateParams(
        sub_property_1='sub_property_1_value_1')
    example_sub_state_obj = ExampleSubStateControlledClass(
        init_from_params=example_sub_state_params)
    example_obj.sub_state = example_sub_state_obj
    assert example_obj.sub_state is not None
    assert example_obj.sub_state.sub_property_1 == 'sub_property_1_value_1'
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None,
        sub_state=ExampleSubState(
            sub_property_1='sub_property_1_value_1'))

  def test_set_property_value_without_triggering_getters(self):
    example_obj = ExampleStateControlledClass()
    example_obj.sub_state_getter_called = False

    example_sub_state_params = ExampleSubStateParams(
        sub_property_1='sub_property_1_value_1')
    example_sub_state_obj = ExampleSubStateControlledClass(
        init_from_params=example_sub_state_params)
    example_obj.set_property_value_without_triggering_getters(
        'sub_state',
        example_sub_state_obj)
    assert example_obj.sub_state_getter_called == False

    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None,
        sub_state=ExampleSubState(
            sub_property_1='sub_property_1_value_1'))
    assert example_obj.sub_state.sub_property_1 == 'sub_property_1_value_1'
    assert example_obj.sub_state_getter_called == True

  def test_load_state(self):
    example_sub_state_params = ExampleSubStateParams(
        sub_property_1='sub_property_1_value_1')
    example_sub_state_obj = ExampleSubStateControlledClass(
        init_from_params=example_sub_state_params)
    example_obj = ExampleStateControlledClass(
        init_from_params=ExampleStateParams(
            sub_state=example_sub_state_obj))
    assert example_obj.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None,
        sub_state=ExampleSubState(
            sub_property_1='sub_property_1_value_1'))

    example_obj_2 = ExampleStateControlledClass()
    example_obj_2.load_state(example_obj.get_state())
    assert example_obj_2.get_internal_state() == ExampleState(
        property_1=None,
        property_2=None,
        property_3=None,
        sub_state=ExampleSubState(
            sub_property_1='sub_property_1_value_1'))
    assert example_obj_2.sub_state.sub_property_1 == 'sub_property_1_value_1'

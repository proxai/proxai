import pytest
import proxai.state_controllers.state_controller as state_controller
from dataclasses import dataclass


@dataclass
class BaseStateConfigClass:
  test_property_1: str
  test_property_2: str
  test_property_3: str


class BaseStateClass(BaseStateConfigClass):
  def __init__(self):
    self._test_property_1 = None
    self._test_property_2 = None
    self._test_property_3 = None
    self._state = BaseStateConfigClass(
        test_property_1=None,
        test_property_2=None,
        test_property_3=None)


class TestStateController(state_controller.StateController):
  @classmethod
  def get_internal_state_property_name(cls):
    return '_state'


class TestInternalStateController:
  def test_update_internal_state_getter_decorator(self):
    class TestClass(BaseStateClass):
      @property
      @TestStateController.getter
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      def test_property_1(self, value):
        self._test_property_1 = value

    test_obj = TestClass()
    assert test_obj._state.test_property_1 is None

    test_obj._test_property_1 = 'test'
    assert test_obj._state.test_property_1 is None

    # Should update state when property accessed
    _ = test_obj.test_property_1
    assert test_obj._state.test_property_1 == 'test'

    # Should not update state when property set
    test_obj.test_property_1 = 'test2'
    assert test_obj._state.test_property_1 == 'test'

  def test_update_internal_state_setter_decorator(self):
    class TestClass(BaseStateClass):
      @property
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      @TestStateController.setter
      def test_property_1(self, value):
        self._test_property_1 = value

    test_obj = TestClass()
    assert test_obj._state.test_property_1 is None

    test_obj._test_property_1 = 'test'
    assert test_obj._state.test_property_1 is None

    # Should not update state when property accessed
    _ = test_obj.test_property_1
    assert test_obj._state.test_property_1 is None

    # Should update state when property set
    test_obj.test_property_1 = 'test2'
    assert test_obj._state.test_property_1 == 'test2'

  def test_update_state_with_multiple_properties(self):
    class TestClass(BaseStateClass):
      @property
      @TestStateController.getter
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      @TestStateController.setter
      def test_property_1(self, value):
        self._test_property_1 = value

      @property
      @TestStateController.getter
      def test_property_2(self):
        return self._test_property_2

      @test_property_2.setter
      @TestStateController.setter
      def test_property_2(self, value):
        self._test_property_2 = value

    test_obj = TestClass()
    assert test_obj._state.test_property_1 is None
    assert test_obj._state.test_property_2 is None

    test_obj.test_property_1 = 'test1'
    assert test_obj._state.test_property_1 == 'test1'
    assert test_obj._state.test_property_2 is None


    test_obj.test_property_2 = 'test2'
    assert test_obj._state.test_property_1 == 'test1'
    assert test_obj._state.test_property_2 == 'test2'

    test_obj._test_property_1 = 'test3'
    assert test_obj._state.test_property_1 == 'test1'
    assert test_obj._state.test_property_2 == 'test2'
    assert test_obj.test_property_1 == 'test3'
    assert test_obj.test_property_2 == 'test2'

    test_obj._test_property_2 = 'test4'
    assert test_obj._state.test_property_1 == 'test3'
    assert test_obj._state.test_property_2 == 'test2'
    assert test_obj.test_property_1 == 'test3'
    assert test_obj.test_property_2 == 'test4'


class TestDependencyStateController:
  def test_requires_dependencies_decorator(self):
    class TestClass(BaseStateClass):
      @property
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      def test_property_1(self, value):
        self._test_property_1 = value

      @property
      def test_property_2(self):
        return self._test_property_2

      @test_property_2.setter
      @TestStateController.requires_dependencies(['test_property_1'])
      def test_property_2(self, value):
        self._test_property_2 = value

    test_obj = TestClass()

    # Should raise error when dependency not set
    with pytest.raises(ValueError):
      test_obj.test_property_2 = 'test'

    # Should work when dependency is set
    test_obj.test_property_1 = 'dependency'
    test_obj.test_property_2 = 'test'
    assert test_obj.test_property_2 == 'test'

  def test_requires_multiple_dependencies_decorator(self):
    class TestClass(BaseStateClass):
      @property
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      def test_property_1(self, value):
        self._test_property_1 = value

      @property
      def test_property_2(self):
        return self._test_property_2

      @test_property_2.setter
      def test_property_2(self, value):
        self._test_property_2 = value


      @property
      def test_property_3(self):
        return self._test_property_3

      @test_property_3.setter
      @TestStateController.requires_dependencies(
          ['test_property_1', 'test_property_2'])
      def test_property_3(self, value):
        self._test_property_3 = value

    test_obj = TestClass()

    # Should raise error when no dependencies set
    with pytest.raises(ValueError):
      test_obj.test_property_3 = 'test'

    # Should raise error when only first dependency set
    test_obj.test_property_1 = 'dep1'
    with pytest.raises(ValueError):
      test_obj.test_property_3 = 'test'

    # Should work when all dependencies set
    test_obj.test_property_2 = 'dep2'
    test_obj.test_property_3 = 'test'
    assert test_obj.test_property_3 == 'test'

  def test_requires_dependencies_decorator_with_none_value(self):
    class TestClass(BaseStateClass):
      @property
      def test_property_1(self):
        return self._test_property_1

      @test_property_1.setter
      def test_property_1(self, value):
        self._test_property_1 = value

      @property
      def test_property_2(self):
        return self._test_property_2

      @test_property_2.setter
      @TestStateController.requires_dependencies(['test_property_1'])
      def test_property_2(self, value):
        self._test_property_2 = value

    test_obj = TestClass()
    test_obj.test_property_2 = None
    with pytest.raises(ValueError):
      test_obj.test_property_2 = 'test1'
    test_obj.test_property_1 = 'test2'
    test_obj.test_property_2 = 'test3'
    test_obj.test_property_1 = None
    _ = test_obj.test_property_2
    with pytest.raises(ValueError):
      test_obj.test_property_2 = 'test4'

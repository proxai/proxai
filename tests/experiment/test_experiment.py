import pytest

import proxai.experiment.experiment as experiment


class TestExperiment:
  def test_invalid_empty_path(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('')

  def test_invalid_path_with_slash(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('/')

  def test_invalid_path_start_with_slash(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('/root')

  def test_invalid_path_end_with_slash(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/')

  def test_invalid_path_with_double_slash(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root//dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root///dir')

  def test_invalid_path_with_double_dot(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('..')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/..')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/../dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('../dir')

  def test_invalid_path_with_tilde(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('~')

  def test_invalid_path_with_dot(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('.')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/.')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/./dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('./dir')

  def test_valid_path(self):
    experiment.validate_experiment_path('root')
    experiment.validate_experiment_path('(root)')
    experiment.validate_experiment_path('root/dir')
    experiment.validate_experiment_path('(root)/dir')
    experiment.validate_experiment_path(
        'root/dir/dir-2/dir_3/dir.dir/dir:3/(dir)(dir)/dir')

  def test_invalid_none_path(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path(None)

  def test_invalid_path_with_spaces(self):
    # Leading/trailing spaces in components are not allowed
    with pytest.raises(ValueError):
      experiment.validate_experiment_path(' root')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root ')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/ dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/dir ')

    # Consecutive spaces are not allowed
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root  dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/my  dir')

  def test_invalid_path_with_special_chars(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root$dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root@/dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root#dir')

  def test_invalid_path_with_unicode(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/测试')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('実験/dir')

  def test_invalid_very_long_path(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('a' * 550)
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/' + 'a' * 120)

  def test_invalid_mixed_slashes(self):
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root\\dir')
    with pytest.raises(ValueError):
      experiment.validate_experiment_path('root/dir\\subdir')

  def test_valid_path_with_spaces(self):
    # Single spaces within components should now be valid
    experiment.validate_experiment_path('root dir')
    experiment.validate_experiment_path('my root/test dir/final dir')
    experiment.validate_experiment_path('root/my dir/test')

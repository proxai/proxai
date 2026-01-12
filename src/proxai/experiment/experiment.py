import datetime
import random
import string

MAX_COMPONENT_LENGTH = 64
MAX_PATH_LENGTH = 500
RESERVED_NAMES = {
    'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4',
    'com5', 'com6', 'com7', 'com8', 'com9', 'lpt1', 'lpt2',
    'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
}

def get_hidden_run_key() -> str:
  """Generate a random run key for identifying unique sessions."""
  random.seed(datetime.datetime.now().timestamp())
  return str(random.randint(1, 1000000))


def validate_experiment_path(experiment_path: str):
  """Validate an experiment path for safe filesystem usage."""
  if not experiment_path:
    raise ValueError('Experiment path cannot be empty')

  if len(experiment_path) > MAX_PATH_LENGTH:
    raise ValueError(
      f'Experiment path cannot be longer than {MAX_PATH_LENGTH} characters')

  allowed_chars = set(string.ascii_letters + string.digits + '()_-.:/ ')
  for char in experiment_path:
    if char not in allowed_chars:
      raise ValueError(
          'Experiment path can only contain following characters:\n'
          f'{sorted(allowed_chars)}')
  if experiment_path.startswith('/'):
    raise ValueError('Experiment path cannot start with "/"')
  if experiment_path.endswith('/'):
    raise ValueError('Experiment path cannot end with "/"')
  if '//' in experiment_path:
    raise ValueError('Experiment path cannot contain "//"')
  if '..' in experiment_path:
    raise ValueError('Experiment path cannot contain ".."')
  if '  ' in experiment_path:
    raise ValueError('Experiment path cannot contain consecutive spaces')
  for component in experiment_path.split('/'):
    if len(component) > MAX_COMPONENT_LENGTH:
      raise ValueError(
        f'Path component "{component}" exceeds maximum length of '
        f'{MAX_COMPONENT_LENGTH}')
    if component == '.':
      raise ValueError('Filename in experiment path cannot be "."')
    if component == '-':
      raise ValueError('Filename in experiment path cannot be "-"')
    if component == '_':
      raise ValueError('Filename in experiment path cannot be "_"')
    if component.lower() in RESERVED_NAMES:
      raise ValueError(f'Path component "{component}" is a reserved name')
    if component.startswith(' ') or component.endswith(' '):
      raise ValueError(
          f'Path component "{component}" cannot start or end with spaces')

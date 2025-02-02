import random
import datetime
import string


def get_hidden_run_key() -> str:
  random.seed(datetime.datetime.now().timestamp())
  return str(random.randint(1, 1000000))


def validate_experiment_path(experiment_path: str):
  allowed_chars = set(string.ascii_letters + string.digits + ' ()_-.:/')
  for char in experiment_path:
    if char not in allowed_chars:
      raise ValueError(
          'Experiment path can only contain following characters:\n'
          f'{sorted(list(allowed_chars))}')

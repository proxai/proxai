import random
import datetime
import string


def get_hidden_run_key() -> str:
  random.seed(datetime.datetime.now().timestamp())
  return str(random.randint(1, 1000000))


def validate_experiment_name(experiment_name: str):
  allowed_chars = set(string.ascii_letters + string.digits + ' ()_-.:/')
  for char in experiment_name:
    if char not in allowed_chars:
      raise ValueError(
          'Experiment name can only contain following characters:\n'
          f'{sorted(list(allowed_chars))}')

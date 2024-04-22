"""Examples of asking about model properties."""
import csv
import os
from pathlib import Path
from typing import List
import json
import datetime
import enum
import dataclasses
import proxai as px
import proxai.types as px_types

# https://www.kaggle.com/datasets/thedevastator/mathematical-problems-dataset-various-mathematic
TEST_DATA = [
  {'answer': '5',
   'question': 'Suppose 12*q + 42 = q + 42. Solve q = 22*t - 0*t - 110 for t.'},
  {'answer': '-4',
   'question': 'Suppose -3*p = -p + m + 60, 3*m + 151 = -5*p. Let u(d) = 2*d**2 '
               '+ 58*d + 9. Let s be u(p). Solve 3*j = -s - 3 for j.'},
  {'answer': '4',
   'question': 'Suppose 8*b - 6*b - 2*q = 16, 0 = q + 5. Let k be (12 - 7 - 8) '
               '+ 3. Suppose h - 32 = -h + 2*g, -5*g - 20 = 0. Solve b*s + k*s '
               '= h for s.'},
  {'answer': '0',
   'question': 'Let t = -10 - -10. Let v(j) = j**2 + 53*j + 285. Let g be '
               'v(-47). Solve 0 = -t*k - g*k for k.'},
  {'answer': '-3',
   'question': 'Let t(n) = -n + 2. Suppose 4*c = 5*b, -7*c + 5*b = -9*c. Let a '
               'be t(c). Solve -a*d - 2*d = 12 for d.'},
  {'answer': '-3',
   'question': 'Let f(k) be the first derivative of -k**3/3 + 2*k**2 - k + 18. '
               'Let y be f(3). Suppose -y = 2*r - 2*q, 3*r - 3*q = 2*r - 11. '
               'Solve -1 + 13 = -r*m for m.'},
  {'answer': '-2',
   'question': 'Let w = -4494 + 4515. Solve 3*o + w = 15 for o.'},
  {'answer': '3',
   'question': 'Let r(h) = h**3 + 22*h**2 - 26*h - 59. Let d be r(-23). Solve 6 '
               '= -8*t + d*t for t.'},
  {'answer': '3',
   'question': 'Let w(u) = -2763*u - 13815. Let v be w(-5). Let g(q) = q**3 - '
               '4*q**2 + 2*q + 3. Let m be g(3). Solve v = -m*y - y + 3 for y.'},
  {'answer': '2',
   'question': 'Let q = 123 - 119. Suppose -q*u - 5*i + 5 = 0, -u - 8 = -5*i - '
               '3. Solve u = 3*z - 5*z + 4 for z.'},
  {'answer': '-1',
   'question': 'Let j(k) = 3*k**2 + 7*k - 12. Let t be j(2). Let q = -12 + t. '
               'Solve q*g - 2 = -4 for g.'},
  {'answer': '3',
   'question': 'Suppose 2*r - 5*p + 47 = 0, r + 5*p = -7 - 24. Let u = -24 - r. '
               'Solve -z - u*z = -9 for z.'}
]


def connect_to_proxai():
  cache_path = f'{Path.home()}/proxai_cache/'
  logging_path = f'{Path.home()}/proxai_log/math_problems/'
  os.makedirs(cache_path, exist_ok=True)
  os.makedirs(logging_path, exist_ok=True)
  px.connect(cache_path=cache_path, logging_path=logging_path)


def get_models(verbose=True):
  models = px.models.generate_text(verbose=True)
  if verbose:
    print('Available models:')
    for provider, provider_models in models.items():
      print(f'{provider}:')
      for provider_model in provider_models:
        print(f'   {provider_model}')
    print()
  return models


def get_answer(question):
  return px.generate_text(f"""\
Can you give me exactly one integer answer to the following question? \
Nothing else, just the answer.
Question: {question}
""")


@dataclasses.dataclass
class QuestionResult:
  correct: bool
  incorrect: int = 0
  error: int = 0


def get_result_for_question(question, answer, try_count) -> QuestionResult:
  question_result = QuestionResult(correct=False)
  for _ in range(try_count):
    try:
      result = get_answer(question)
      if result == answer:
        question_result.correct = True
        return question_result
      question_result.incorrect += 1
    except Exception as e:
      question_result.error += 1
  return question_result


@dataclasses.dataclass
class EvalResult:
  correct: int = 0
  incorrect: int = 0
  error: int = 0
  all_results: List[str] = dataclasses.field(default_factory=list)


def eval_math_questions(try_count) -> EvalResult:
  eval_result = EvalResult()
  for idx, test in enumerate(TEST_DATA):
    # print(f'{idx+1}/{len(TEST_DATA)}')
    question_result = get_result_for_question(
        question=test['question'],
        answer=test['answer'],
        try_count=try_count)
    if question_result.correct:
      eval_result.correct += 1
      eval_result.all_results.append('True')
    else:
      if question_result.error == try_count:
        eval_result.error += 1
        eval_result.all_results.append('Error')
      else:
        eval_result.incorrect += 1
        eval_result.all_results.append('False')
  return eval_result


def run_test(models, try_count):
  print(f'{"PROVIDER":15} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  print()
  all_results = {}
  for provider in sorted(list(models.keys())):
    provider_models = sorted(models[provider])
    for provider_model in provider_models:
      px.set_model(generate_text=(provider, provider_model))
      start_time = datetime.datetime.now()
      # print(f'{provider:10} | {provider_model:35}')
      eval_result = eval_math_questions(try_count=try_count)
      end_time = datetime.datetime.now()
      duration = (end_time - start_time).total_seconds() * 1000
      response = (
        f'Correct: {eval_result.correct:2}, '
        f'Incorrect: {eval_result.incorrect:2}, '
        f'Error: {eval_result.error:2}')
      print(f'{provider:15} | {provider_model:45} | {duration:10.0f} ms | {response}')
      all_results[(provider, provider_model)] = eval_result.all_results
  print()
  return all_results


def print_all_results(all_results):
  print(f'{"MODEL":60} | '
        + ' | '.join([f'{idx:5}' for idx in range(1, len(TEST_DATA)+1)]),
        end=' | ')
  print()
  for (provider, provider_model), results in all_results.items():
    print(f"{f'{provider} / {provider_model}':60} | ", end='')
    for result in results:
      print(f'{result:5}', end=' | ')
    print()
  print(f"{'Total count':60} | ", end='')
  for idx in range(1, len(TEST_DATA)+1):
    true_count = sum([1 for results in all_results.values() if results[idx-1] == 'True'])
    print(f'{true_count:5}', end=' | ')
  print()


def main():
  _TRY_COUNT = 3
  connect_to_proxai()
  models = get_models()
  all_results = run_test(models, _TRY_COUNT)
  print_all_results(all_results)


if __name__ == '__main__':
  main()

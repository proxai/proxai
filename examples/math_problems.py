"""Examples of asking about model properties."""
import csv
import json
import datetime
import proxai
import proxai.types as types

# https://www.kaggle.com/datasets/thedevastator/mathematical-problems-dataset-various-mathematic
test_data = [
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
               'Solve -z - u*z = -9 for z.'}]


def _log_result(row):
  with open('/Users/osmanaka/temp/math_problems_log.jsonl', 'a') as f:
    f.write(json.dumps(row) + '\n')
  f.close()
  return

def get_answer(question):
  return proxai.generate_text(f"""\
Can you give me exactly one integer answer to the following question? \
Nothing else, just the answer.
Question: {question}
""")


def eval_math_questions(provider, model, try_count=3):
  correct_answers = 0
  incorrect_answers = 0
  model_error = 0
  for idx, test in enumerate(test_data):
    correct_flag = False
    error_count = 0
    for try_id in range(try_count):
      row = {
        'provider': provider,
        'model': model,
        'question': test['question'],
        'answer': test['answer'],
        'response': '',
        'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'success': '',
      }
      try:
        result = get_answer(test['question'])
        row['response'] = result
        if result == test['answer']:
          row['success'] = 'CORRECT'
          _log_result(row)
          correct_flag = True
          break
        else:
          row['success'] = 'INCORRECT'
          _log_result(row)
          error_count += 1
      except Exception as e:
        row['response'] = str(e)
        row['success'] = 'MODEL ERROR'
        _log_result(row)
        error_count += 1
    if correct_flag:
      correct_answers += 1
    elif error_count != try_count:
      incorrect_answers += 1
    else:
      model_error += 1

  return {
    'correct_answers': correct_answers,
    'incorrect_answers': incorrect_answers,
    'model_error': model_error
  }

def main():
  print(f'{"PROVIDER":10} | {"MODEL":35} | {"DURATION":13} | {"RESPONSE"}')
  print()
  for provider, models in types._MODEL_MAP.items():
    for model_name in models:
      proxai.register_model(provider, model_name)
      start_time = datetime.datetime.now()
      response = eval_math_questions(provider, model_name)
      end_time = datetime.datetime.now()
      duration = (end_time - start_time).total_seconds() * 1000
      response = (
        f'Correct: {response["correct_answers"]:2}, '
        f'Incorrect: {response["incorrect_answers"]:2}, '
        f'Model Error: {response["model_error"]:2}')
      print(f'{provider:10} | {model_name:35} | {duration:10.0f} ms | {response}')
  print()


if __name__ == '__main__':
  main()

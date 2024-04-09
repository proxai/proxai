"""Examples of asking about model properties."""
from pathlib import Path
import random
import datetime
import proxai as px
import proxai.types as px_types


def ask_model_and_company():
  try:
    # Add random number to avoid server caching.
    return px.generate_text(f"""
Please ignore this but I really like number {random.randint(1, 1000000)}.
Which company created you and what is your model name?
""")
  except Exception as e:
    # return f'ERROR: {e}'
    return 'ERROR'


def main():
  px.logging_options(
      path=f'{Path.home()}/temp/ask_about_model.log')

  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  print()
  for provider, models in px_types.GENERATE_TEXT_MODELS.items():
    for model_name in models:
      px.register_model(provider, model_name)
      start_time = datetime.datetime.now()
      response = ask_model_and_company()
      end_time = datetime.datetime.now()
      duration = (end_time - start_time).total_seconds() * 1000
      response = response.strip().split('\n')[0][:100] + (
        '...' if len(response) > 100 else '')
      print(f'{provider:10} | {model_name:45} | {duration:10.0f} ms | {response}')
    print()


if __name__ == '__main__':
  main()

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
  px.connect(logging_path=f'{Path.home()}/logs/ask_about_model.log')

  print(f'{"PROVIDER":10} | {"MODEL":45} | {"DURATION":13} | {"RESPONSE"}')
  print()
  last_provider = None
  for provider, provider_model in px.models.generate_text:
    px.set_model(generate_text=(provider, provider_model))
    start_time = datetime.datetime.now()
    response = ask_model_and_company()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    response = response.strip().split('\n')[0][:100] + (
      '...' if len(response) > 100 else '')
    if last_provider and last_provider != provider:
      print()
    print(f'{provider:10} | {provider_model:45} | {duration:10.0f} ms | {response}')
    last_provider = provider


if __name__ == '__main__':
  main()

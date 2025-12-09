import copy
import proxai as px
import proxai.types as px_types
from pydantic import BaseModel

PROMPT = f'A=5, B=10. Give me {{"A": A, "B": B, "C": A + B}}.'

MESSAGES = [
  {"role": "user", "content": PROMPT}
]

JSON_SCHEMA = {
  "type": "json_schema",
  "json_schema": {
    "name": "SumOfNumbers",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "A": {
          "type": "integer",
          "description": "The value of A."
        },
        "B": {
          "type": "integer",
          "description": "The value of B."
        },
        "C": {
          "type": "number",
          "description": "Calculated: A + B"
        }
      },
      "required": ["A", "B", "C"],
      "additionalProperties": False
    }
  }
}

class SumOfNumbers(BaseModel):
  A: int
  B: int
  C: int

TEST_FEATURES = {
    'messages': MESSAGES,
    'system': 'You are a helpful assistant.',
    'max_tokens': 1000,
    'temperature': 0.5,
    # 'stop': '\n\n', # Fails for Claude
    'stop': 'STOP',
    'response_format::text': px.ResponseFormat(type=px.ResponseFormatType.TEXT),
    'response_format::json': px.ResponseFormat(type=px.ResponseFormatType.JSON),
    'response_format::json_schema': px.ResponseFormat(type=px.ResponseFormatType.JSON_SCHEMA, schema=JSON_SCHEMA),
    'response_format::pydantic': px.ResponseFormat(type=px.ResponseFormatType.PYDANTIC, schema=SumOfNumbers),
    'web_search': True,
  }


def test_feature_compatibility(
    provider_model: px_types.ProviderModelIdentifierType,
    feature_mapping_strategy: px_types.FeatureMappingStrategy,
    put_response_format_to_supported: bool,
):
  print('=================================', provider_model)
  print('=================================', feature_mapping_strategy)
  print('=================================', put_response_format_to_supported)
  px.reset_state()
  px.connect(
    feature_mapping_strategy=feature_mapping_strategy)

  model_configs = px._get_model_configs()
  config = copy.deepcopy(model_configs.model_configs_schema)
  provider_model_features = config.version_config.provider_model_configs[
      provider_model[0]][provider_model[1]].features
  provider_model_features.supported = [
      'messages',
      'system',
      'max_tokens',
      'temperature',
      'stop',
      'response_format::text',
  ]
  provider_model_features.best_effort = []

  if put_response_format_to_supported:
    provider_model_features.supported.extend([
        'response_format::json',
        'response_format::json_schema',
        'response_format::pydantic',
    ])
  else:
    provider_model_features.best_effort.extend([
        'response_format::json',
        'response_format::json_schema',
        'response_format::pydantic',
    ])

  model_configs.model_configs_schema = config

  print('------- Plain call:')
  try:
    response = px.generate_text(
        PROMPT,
        provider_model=provider_model)
    print('SUCCESS: ', response)
  except Exception as e:
    print('ERROR: ', e)
    input('Press Enter to continue...')

  print('------- System call:')
  try:
    response = px.generate_text(
        PROMPT,
        provider_model=provider_model,
        system=TEST_FEATURES['system'])
    print('SUCCESS: ', response)
  except Exception as e:
    print('ERROR: ', e)
    input('Press Enter to continue...')

  print('------- Response format JSON call:')
  try:
    response = px.generate_text(
      PROMPT,
        provider_model=provider_model,
        response_format=TEST_FEATURES['response_format::json'])
    print('SUCCESS: ', response)
  except Exception as e:
    print('ERROR: ', e)
    input('Press Enter to continue...')

  print('------- Response format JSON Schema call:')
  try:
    response = px.generate_text(
      PROMPT,
      provider_model=provider_model,
      response_format=TEST_FEATURES['response_format::json_schema'])
    print('SUCCESS: ', response)
  except Exception as e:
    print('ERROR: ', e)
    input('Press Enter to continue...')

  print('------- Response format Pydantic call:')
  try:
    response = px.generate_text(
      PROMPT,
      provider_model=provider_model,
        response_format=TEST_FEATURES['response_format::pydantic'])
    print('SUCCESS: ', response)
  except Exception as e:
    print('ERROR: ', e)
    input('Press Enter to continue...')


def main():
  # provider_model = ('openai', 'gpt-4o-mini')
  # provider_model = ('openai', 'gpt-5.1')
  # provider_model = ('claude', 'haiku-4.5')
  # provider_model = ('claude', 'haiku-3')
  # provider_model = ('gemini', 'gemini-2.5-flash')
  provider_model = ('grok', 'grok-3-fast-beta')
  # provider_model = ('mistral', 'mistral-small')

  test_feature_compatibility(
      provider_model=provider_model,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      put_response_format_to_supported=True)

  test_feature_compatibility(
      provider_model=provider_model,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      put_response_format_to_supported=False)

  test_feature_compatibility(
      provider_model=provider_model,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.BEST_EFFORT,
      put_response_format_to_supported=True)

  test_feature_compatibility(
      provider_model=provider_model,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.BEST_EFFORT,
      put_response_format_to_supported=False)


if __name__ == '__main__':
  main()

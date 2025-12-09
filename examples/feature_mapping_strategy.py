import copy
import proxai as px
import proxai.types as px_types
from pydantic import BaseModel

PROMPT = f'A=5, B=10. Give me {{"A": A, "B": B, "C": A + B}}.'

MESSAGES = [
  {"role": "user", "content": PROMPT}
]

JSON_SCHEMA = {
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


def simple_test():
  # provider_model = ('openai', 'gpt-4o-mini')
  # provider_model = ('openai', 'gpt-5.1')
  # provider_model = ('claude', 'haiku-4.5')
  provider_model = ('claude', 'haiku-3')

  px.connect(
    feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT)

  model_configs = px._get_model_configs()
  config = copy.deepcopy(model_configs.model_configs_schema)
  config.version_config.provider_model_configs[
      provider_model[0]][provider_model[1]].features.supported = [
          'messages',
          'system',
          'max_tokens',
          'temperature',
          'stop',
          'response_format::text',
          'response_format::json',
          'response_format::json_schema',
      ]
  config.version_config.provider_model_configs[
      provider_model[0]][provider_model[1]].features.best_effort = [
          'response_format::pydantic',
      ]
  model_configs.model_configs_schema = config

  # response = px.generate_text(
  #     PROMPT,
  #     provider_model=provider_model)

  # response = px.generate_text(
  #     PROMPT,
  #     provider_model=provider_model,
  #     system=TEST_FEATURES['system'])

  response = px.generate_text(
      PROMPT,
      provider_model=provider_model,
      response_format=TEST_FEATURES['response_format::pydantic'])

  print(response)


if __name__ == '__main__':
  simple_test()

import copy
import traceback
from pprint import pprint
import functools
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
    'prompt': PROMPT,
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


def get_feature_config(
    features: list[str],
    config_features: list[str],
    endpoint: str,
    support_type: str,
):
  result_config = {}
  for feature in config_features:
    result_config[feature] = px_types.EndpointFeatureInfoType(
        supported=[endpoint],
        best_effort=[],
        not_supported=[],
    )

  for feature in features:
    if feature not in config_features:
      result_config[feature] = px_types.EndpointFeatureInfoType(
          supported=[endpoint] if support_type == 'supported' else [],
          best_effort=[endpoint] if support_type == 'best_effort' else [],
          not_supported=[endpoint] if support_type == 'not_supported' else [],
      )

  return result_config


def setup_connection_with_feature_and_endpoint(
    provider_model: px_types.ProviderModelIdentifierType,
    feature_mapping_strategy: px_types.FeatureMappingStrategy,
    features: list[str],
    config_features: list[str],
    endpoint: str,
    support_type: str,
):
  px.reset_state()
  px.connect(feature_mapping_strategy=feature_mapping_strategy)

  model_configs = px._get_model_configs()
  config = copy.deepcopy(model_configs.model_configs_schema)
  provider_model_config = config.version_config.provider_model_configs[
      provider_model[0]][provider_model[1]]
  feature_config = get_feature_config(
      features=features,
      config_features=config_features,
      endpoint=endpoint,
      support_type=support_type,
  )
  provider_model_config.features = feature_config
  model_configs.model_configs_schema = config
  return feature_config


def get_test_feature_call_func(
    provider_model: px_types.ProviderModelIdentifierType,
    features: list[str],
):
  call_func = functools.partial(px.generate_text, provider_model=provider_model)

  for feature in features:
    if feature == 'prompt':
      call_func = functools.partial(
          call_func,
          prompt=TEST_FEATURES['prompt'])
    elif feature == 'messages':
      call_func = functools.partial(
          call_func,
          messages=TEST_FEATURES['messages'])
    elif feature == 'system':
      call_func = functools.partial(
          call_func,
          system=TEST_FEATURES['system'])
    elif feature == 'max_tokens':
      call_func = functools.partial(
          call_func,
          max_tokens=TEST_FEATURES['max_tokens'])
    elif feature == 'temperature':
      call_func = functools.partial(
          call_func,
          temperature=TEST_FEATURES['temperature'])
    elif feature == 'stop':
      call_func = functools.partial(
          call_func,
          stop=TEST_FEATURES['stop'])
    elif feature == 'response_format::text':
      call_func = functools.partial(
          call_func,
          response_format=TEST_FEATURES['response_format::text'])
    elif feature == 'response_format::json':
      call_func = functools.partial(
          call_func,
          response_format=TEST_FEATURES['response_format::json'])
    elif feature == 'response_format::json_schema':
      call_func = functools.partial(
          call_func,
          response_format=TEST_FEATURES['response_format::json_schema'])
    elif feature == 'response_format::pydantic':
      call_func = functools.partial(
          call_func,
          response_format=TEST_FEATURES['response_format::pydantic'])
    elif feature == 'web_search':
      call_func = functools.partial(
          call_func,
          web_search=TEST_FEATURES['web_search'])

  return call_func

def test_feature_compatibility(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoint: str,
    feature_mapping_strategy: px_types.FeatureMappingStrategy,
    features: str,
    config_features: str,
    support_type: str,
    verbose: bool = True,
):
  if verbose:
    print(f'---------------------------------')
    print(f'> Provider model           : {provider_model}')
    print(f'> Endpoint                 : {endpoint}')
    print(f'> Feature mapping strategy : {feature_mapping_strategy}')
    print(f'> Features                 : {features}')
    print(f'> Config features          : {config_features}')
    print(f'> Support type             : {support_type}')
  feature_config = setup_connection_with_feature_and_endpoint(
      provider_model=provider_model,
      feature_mapping_strategy=feature_mapping_strategy,
      features=features,
      config_features=config_features,
      endpoint=endpoint,
      support_type=support_type,
  )
  try:
    response = get_test_feature_call_func(
        provider_model=provider_model,
        features=features,
    )()
    if verbose:
      print(f'SUCCESS: {type(response)}:\n{response}')
    return True
  except Exception as e:
    if verbose:
      print('ERROR: ', e)
      print(traceback.format_exc())
      print('FEATURE CONFIG:')
      pprint(feature_config)
      input('Press Enter to continue...')
    return False


def find_endpoint_minimum_required_features(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoint: str,
    verbose: bool = True,
):
  test_required_features = [
      ['prompt', 'response_format::text'],
      ['prompt', 'response_format::pydantic'],
      ['prompt', 'max_tokens', 'response_format::text'],
      ['prompt', 'max_tokens', 'response_format::pydantic'],
  ]
  for required_features in test_required_features:
    if test_feature_compatibility(
        provider_model=provider_model,
        endpoint=endpoint,
        feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
        features=required_features,
        config_features=required_features,
        support_type='supported',
        verbose=verbose):
      return required_features

  raise ValueError(
      f'No response format feature found for {endpoint} and {provider_model}'
      '\nThis is probably a bug in the provider implementation of ProxAI.')


def get_updated_feature_list(
  test_feature: str,
  required_features: list[str],
):
  result_features = copy.deepcopy(required_features)
  if test_feature == 'messages' and 'prompt' in result_features:
    result_features.remove('prompt')

  if test_feature == 'prompt' and 'messages' in result_features:
    result_features.remove('messages')

  if test_feature.startswith('response_format::'):
    if 'response_format::text' in result_features:
      result_features.remove('response_format::text')
    if 'response_format::json' in result_features:
      result_features.remove('response_format::json')
    if 'response_format::json_schema' in result_features:
      result_features.remove('response_format::json_schema')
    if 'response_format::pydantic' in result_features:
      result_features.remove('response_format::pydantic')

  return result_features + [test_feature]


def generate_config_for_model(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoints: list[str],
    test_features: list[str],
    verbose: bool = True,
    verbose_test_feature_compatibility: bool = False,
    verbose_find_endpoint_minimum_required_features: bool = False,
):
  result_configs = {}
  endpoint_minimum_required_features = {}
  for endpoint in endpoints:
    endpoint_minimum_required_features[
        endpoint] = find_endpoint_minimum_required_features(
            provider_model=provider_model,
            endpoint=endpoint,
            verbose=verbose_find_endpoint_minimum_required_features)
  if verbose:
    print(f'Endpoint minimum required features:')
    pprint(endpoint_minimum_required_features)
  for test_feature in test_features:
    if verbose:
      print(f'--------------------------------- {test_feature} ---------------------------------')
    result_configs[test_feature] = {
        'supported': [],
        'best_effort': [],
        'not_supported': [],
    }
    for endpoint in endpoints:
      features = get_updated_feature_list(
        test_feature=test_feature,
        required_features=endpoint_minimum_required_features[endpoint],
      )
      config_features = endpoint_minimum_required_features[endpoint]
      if test_feature_compatibility(
          provider_model=provider_model,
          endpoint=endpoint,
          feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
          features=features,
          config_features=config_features,
          support_type='supported',
          verbose=verbose_test_feature_compatibility,
      ):
        result_configs[test_feature]['supported'].append(endpoint)
      elif test_feature_compatibility(
          provider_model=provider_model,
          endpoint=endpoint,
          feature_mapping_strategy=px_types.FeatureMappingStrategy.BEST_EFFORT,
          features=features,
          config_features=config_features,
          support_type='best_effort',
          verbose=verbose_test_feature_compatibility,
      ):
        result_configs[test_feature]['best_effort'].append(endpoint)
      else:
        result_configs[test_feature]['not_supported'].append(endpoint)
    if verbose:
      pprint({test_feature: result_configs[test_feature]})
      input('Press Enter to continue...')
  return result_configs


def main():
  # endpoints = [
  #     'chat.completions.create',
  #     'beta.chat.completions.parse',
  #     'responses.create',
  # ]
  # provider_model = ('openai', 'gpt-4o-mini')
  # provider_model = ('openai', 'gpt-5.1')

  # endpoints = [
  #     'messages.create',
  #     'beta.messages.create',
  #     'beta.messages.parse',
  # ]
  # provider_model = ('claude', 'haiku-3')
  # provider_model = ('claude', 'haiku-4.5')

  endpoints = [
      'models.generate_content',
  ]
  provider_model = ('gemini', 'gemini-2.5-flash')

  # provider_model = ('grok', 'grok-3-fast-beta')
  # provider_model = ('mistral', 'mistral-small')
  # provider_model = ('cohere', 'command-a')
  # provider_model = ('deepseek', 'deepseek-v3')
  # provider_model = ('databricks', 'meta-llama-3-1-8b-it')
  # provider_model = ('huggingface', 'deepseek-v3')

  generate_config_for_model(
      provider_model=provider_model,
      endpoints=endpoints,
      test_features=[
          'prompt',
          'messages',
          'system',
          'max_tokens',
          'temperature',
          'stop',
          'response_format::text',
          'response_format::json',
          'response_format::json_schema',
          'response_format::pydantic',
          'web_search',
      ],
      verbose=True,
      verbose_test_feature_compatibility=False,
      verbose_find_endpoint_minimum_required_features=False,
  )


if __name__ == '__main__':
  main()

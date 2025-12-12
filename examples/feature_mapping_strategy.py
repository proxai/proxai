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
    feature: str,
    response_format_feature: str,
    endpoint: str,
    support_type: str,
):
  result_config = {
    'prompt': px_types.EndpointFeatureInfoType(
        supported=[endpoint],
        best_effort=[],
        not_supported=[],
    )
  }

  result_config[feature] = px_types.EndpointFeatureInfoType(
      supported=[endpoint] if support_type == 'supported' else [],
      best_effort=[endpoint] if support_type == 'best_effort' else [],
      not_supported=[endpoint] if support_type == 'not_supported' else [],
  )

  result_config[response_format_feature] = px_types.EndpointFeatureInfoType(
      supported=[endpoint],
      best_effort=[],
      not_supported=[],
  )
  return result_config


def setup_connection_with_feature_and_endpoint(
    provider_model: px_types.ProviderModelIdentifierType,
    feature_mapping_strategy: px_types.FeatureMappingStrategy,
    feature: str,
    response_format_feature: str,
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
      feature=feature,
      response_format_feature=response_format_feature,
      endpoint=endpoint,
      support_type=support_type,
  )
  provider_model_config.features = feature_config
  model_configs.model_configs_schema = config
  return feature_config


def get_test_feature_call_func(
    provider_model: px_types.ProviderModelIdentifierType,
    feature: str,
    response_format_feature: str,
):
  if feature != 'messages':
    call_func = functools.partial(
      px.generate_text,
      prompt=PROMPT,
      provider_model=provider_model)
  else:
    call_func = functools.partial(
      px.generate_text,
      provider_model=provider_model)

  if not feature.startswith('response_format::'):
    call_func = functools.partial(
      call_func, response_format=TEST_FEATURES[response_format_feature])

  if feature == 'messages':
    call_func = functools.partial(
        call_func, messages=TEST_FEATURES['messages'])
  elif feature == 'system':
    call_func = functools.partial(
      call_func, system=TEST_FEATURES['system'])
  elif feature == 'max_tokens':
    call_func = functools.partial(
      call_func, max_tokens=TEST_FEATURES['max_tokens'])
  elif feature == 'temperature':
    call_func = functools.partial(
      call_func, temperature=TEST_FEATURES['temperature'])
  elif feature == 'stop':
    call_func = functools.partial(
      call_func, stop=TEST_FEATURES['stop'])
  elif feature == 'response_format::text':
    call_func = functools.partial(
      call_func, response_format=TEST_FEATURES['response_format::text'])
  elif feature == 'response_format::json':
    call_func = functools.partial(
      call_func, response_format=TEST_FEATURES['response_format::json'])
  elif feature == 'response_format::json_schema':
    call_func = functools.partial(
      call_func, response_format=TEST_FEATURES['response_format::json_schema'])
  elif feature == 'response_format::pydantic':
    call_func = functools.partial(
      call_func, response_format=TEST_FEATURES['response_format::pydantic'])
  elif feature == 'web_search':
    call_func = functools.partial(
      call_func, web_search=TEST_FEATURES['web_search'])

  return call_func

def test_feature_compatibility(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoint: str,
    feature_mapping_strategy: px_types.FeatureMappingStrategy,
    feature: str,
    response_format_feature: str,
    support_type: str,
    verbose: bool = True,
):
  if verbose:
    print(f'---------------------------------')
    print(f'> Provider model           : {provider_model}')
    print(f'> Endpoint                 : {endpoint}')
    print(f'> Feature mapping strategy : {feature_mapping_strategy}')
    print(f'> Feature                  : {feature}')
    print(f'> Response format feature  : {response_format_feature}')
    print(f'> Support type             : {support_type}')
  feature_config = setup_connection_with_feature_and_endpoint(
      provider_model=provider_model,
      feature_mapping_strategy=feature_mapping_strategy,
      feature=feature,
      response_format_feature=response_format_feature,
      endpoint=endpoint,
      support_type=support_type,
  )
  try:
    response = get_test_feature_call_func(
        provider_model=provider_model,
        feature=feature,
        response_format_feature=response_format_feature
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


def find_endpoints_response_format(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoint: str,
    verbose: bool = True,
):
  if test_feature_compatibility(
      provider_model=provider_model,
      endpoint=endpoint,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      feature='response_format::text',
      response_format_feature='response_format::text',
      support_type='supported',
      verbose=verbose,
  ):
    return 'response_format::text'
  elif test_feature_compatibility(
      provider_model=provider_model,
      endpoint=endpoint,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      feature='response_format::json',
      response_format_feature='response_format::json',
      support_type='supported',
      verbose=verbose,
  ):
    return 'response_format::json'
  elif test_feature_compatibility(
      provider_model=provider_model,
      endpoint=endpoint,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      feature='response_format::json_schema',
      response_format_feature='response_format::json_schema',
      support_type='supported',
      verbose=verbose,
  ):
    return 'response_format::json_schema'
  elif test_feature_compatibility(
      provider_model=provider_model,
      endpoint=endpoint,
      feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
      feature='response_format::pydantic',
      response_format_feature='response_format::pydantic',
      support_type='supported',
      verbose=verbose,
  ):
    return 'response_format::pydantic'
  else:
    raise ValueError(
        f'No response format feature found for {endpoint} and {provider_model}'
        '\nThis is probably a bug in the provider implementation of ProxAI.')

def test_feature_compatibility_v2(
    provider_model: px_types.ProviderModelIdentifierType,
    endpoints: list[str],
    features: list[str],
    verbose: bool = True,
):
  result_configs = {}
  endpoint_response_format_features = {}
  for endpoint in endpoints:
    response_format_feature = find_endpoints_response_format(
        provider_model=provider_model,
        endpoint=endpoint,
        verbose=False,
    )
    endpoint_response_format_features[endpoint] = response_format_feature
  print(f'Endpoint response format features:')
  pprint(endpoint_response_format_features)
  for feature in features:
    print(f'--------------------------------- {feature} ---------------------------------')
    result_configs[feature] = {
        'supported': [],
        'best_effort': [],
        'not_supported': [],
    }
    for endpoint in endpoints:
      if test_feature_compatibility(
          provider_model=provider_model,
          endpoint=endpoint,
          feature_mapping_strategy=px_types.FeatureMappingStrategy.STRICT,
          feature=feature,
          response_format_feature=endpoint_response_format_features[endpoint],
          support_type='supported',
          verbose=verbose,
      ):
        result_configs[feature]['supported'].append(endpoint)
      elif test_feature_compatibility(
          provider_model=provider_model,
          endpoint=endpoint,
          feature_mapping_strategy=px_types.FeatureMappingStrategy.BEST_EFFORT,
          feature=feature,
          response_format_feature=endpoint_response_format_features[endpoint],
          support_type='best_effort',
          verbose=verbose,
      ):
        result_configs[feature]['best_effort'].append(endpoint)
      else:
        result_configs[feature]['not_supported'].append(endpoint)
    pprint({feature: result_configs[feature]})
    input('Press Enter to continue...')
  return result_configs


def main():
  provider_model = ('openai', 'gpt-4o-mini')
  endpoints = [
    'chat.completions.create',
    'beta.chat.completions.parse',
    'responses.create',
  ]
  # provider_model = ('openai', 'gpt-5.1')
  # provider_model = ('claude', 'haiku-4.5')
  # provider_model = ('claude', 'haiku-3')
  # provider_model = ('gemini', 'gemini-2.5-flash')
  # provider_model = ('grok', 'grok-3-fast-beta')
  # provider_model = ('mistral', 'mistral-small')
  # provider_model = ('cohere', 'command-a')
  # provider_model = ('deepseek', 'deepseek-v3')
  # provider_model = ('databricks', 'meta-llama-3-1-8b-it')
  # provider_model = ('huggingface', 'deepseek-v3')

  test_feature_compatibility_v2(
      provider_model=provider_model,
      endpoints=endpoints,
      features=[
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
      verbose=False,
  )


if __name__ == '__main__':
  main()

import datetime
import proxai.types as types
import proxai.stat_types as stat_types
import proxai.serializers.type_serializer as type_serializer
import proxai.serializers.hash_serializer as hash_serializer
import proxai.connectors.model_configs as model_configs
import pytest


def _get_provider_model_type_options():
  return [
      {'provider': 'openai',
       'model': 'gpt-4',
       'provider_model_identifier': 'gpt-4'},]


def _get_provider_model_identifier_options():
  return [
      model_configs.get_provider_model_config(('openai', 'gpt-4')),
      model_configs.get_provider_model_config(('claude', 'opus-4')),
      ('openai', 'gpt-4'),
      ('claude', 'sonnet-4'),]


def _get_provider_model_pricing_type_options():
  return [
      {'per_response_token_cost': 0.001,
       'per_query_token_cost': 0.002},
      {'per_response_token_cost': 0.0,
       'per_query_token_cost': 0.0},
      {'per_response_token_cost': 1.5,
       'per_query_token_cost': 0.5},]


def _get_provider_model_feature_type_options():
  return [
      {},
      {'not_supported_features': []},
      {'not_supported_features': ['feature1']},
      {'not_supported_features': ['feature1', 'feature2', 'feature3']},]


def _get_provider_model_metadata_type_options():
  return [
      {},
      {'call_type': types.CallType.GENERATE_TEXT},
      {'is_featured': True},
      {'is_featured': False},
      {'model_size': types.ModelSizeType.SMALL},
      {'model_size': types.ModelSizeType.MEDIUM},
      {'model_size': types.ModelSizeType.LARGE},
      {'model_size': types.ModelSizeType.LARGEST},
      {'is_default_candidate': True},
      {'is_default_candidate': False},
      {'default_candidate_priority': 1},
      {'default_candidate_priority': 100},
      {'tags': []},
      {'tags': ['tag1']},
      {'tags': ['tag1', 'tag2', 'tag3']},
      {'call_type': types.CallType.GENERATE_TEXT,
       'is_featured': True,
       'model_size': types.ModelSizeType.LARGE,
       'is_default_candidate': True,
       'default_candidate_priority': 5,
       'tags': ['tag1', 'tag2']},]


def _get_provider_model_config_type_options():
  return [
      {},
      {'provider_model': model_configs.get_provider_model_config(('openai', 'gpt-4'))},
      {'pricing': types.ProviderModelPricingType(
          per_response_token_cost=0.001,
          per_query_token_cost=0.002)},
      {'features': types.ProviderModelFeatureType(
          not_supported_features=['feature1'])},
      {'metadata': types.ProviderModelMetadataType(
          call_type=types.CallType.GENERATE_TEXT,
          is_featured=True)},
      {'provider_model': model_configs.get_provider_model_config(('claude', 'opus-4')),
       'pricing': types.ProviderModelPricingType(
          per_response_token_cost=0.003,
          per_query_token_cost=0.001),
       'features': types.ProviderModelFeatureType(
          not_supported_features=['feature1', 'feature2']),
       'metadata': types.ProviderModelMetadataType(
          call_type=types.CallType.GENERATE_TEXT,
          is_featured=True,
          model_size=types.ModelSizeType.LARGEST,
          is_default_candidate=True,
          default_candidate_priority=10,
          tags=['production', 'recommended'])},]


def _get_model_configs_schema_metadata_type_options():
  return [
      {},
      {'version': '1.0.0'},
      {'released_at': datetime.datetime.now(datetime.timezone.utc)},
      {'config_origin': types.ConfigOriginType.BUILT_IN},
      {'config_origin': types.ConfigOriginType.PROXDASH},
      {'release_notes': 'Initial release'},
      {'version': '1.0.0',
       'released_at': datetime.datetime.now(datetime.timezone.utc),
       'config_origin': types.ConfigOriginType.BUILT_IN,
       'release_notes': 'Initial release'},
      {'version': '2.1.0',
       'released_at': datetime.datetime.now(datetime.timezone.utc),
       'config_origin': types.ConfigOriginType.PROXDASH,
       'release_notes': 'Added new models and updated pricing'},]


def _get_model_configs_schema_version_config_type_options():
  return [
      {},
      {'provider_model_configs': {
          'openai': {
              'gpt-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('openai', 'gpt-4')))}}},
      {'provider_model_configs': {
          'openai': {
              'gpt-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('openai', 'gpt-4')),
                  pricing=types.ProviderModelPricingType(
                      per_response_token_cost=0.001,
                      per_query_token_cost=0.002))},
          'claude': {
              'opus-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('claude', 'opus-4')),
                  metadata=types.ProviderModelMetadataType(
                      is_featured=True))}}},
      {'featured_models': {
          'openai': (
              model_configs.get_provider_model_config(('openai', 'gpt-4')),)}},
      {'featured_models': {
          'openai': (
              model_configs.get_provider_model_config(('openai', 'gpt-4')),),
          'claude': (
              ('claude', 'opus-4'),)}},
      {'models_by_call_type': {
          types.CallType.GENERATE_TEXT: {
              'openai': (
                  model_configs.get_provider_model_config(('openai', 'gpt-4')),),
              'claude': (
                  ('claude', 'sonnet-4'),)}}},
      {'models_by_size': {
          types.ModelSizeType.SMALL: (
              ('openai', 'gpt-4o-mini'),),
          types.ModelSizeType.LARGE: (
              model_configs.get_provider_model_config(('openai', 'gpt-4')),
              ('claude', 'opus-4'),)}},
      {'default_model_priority_list': (
          model_configs.get_provider_model_config(('openai', 'gpt-4')),
          ('claude', 'opus-4'),
          ('openai', 'o3-mini'),)},
      {'provider_model_configs': {
          'openai': {
              'gpt-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('openai', 'gpt-4')),
                  pricing=types.ProviderModelPricingType(
                      per_response_token_cost=0.001,
                      per_query_token_cost=0.002),
                  features=types.ProviderModelFeatureType(
                      not_supported_features=['feature1']),
                  metadata=types.ProviderModelMetadataType(
                      call_type=types.CallType.GENERATE_TEXT,
                      is_featured=True,
                      model_size=types.ModelSizeType.LARGE)),
              'o3-mini': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('openai', 'o3-mini')))},
          'claude': {
              'opus-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('claude', 'opus-4')),
                  metadata=types.ProviderModelMetadataType(
                      is_featured=True,
                      model_size=types.ModelSizeType.LARGEST)),
              'sonnet-4': types.ProviderModelConfigType(
                  provider_model=model_configs.get_provider_model_config(
                      ('claude', 'sonnet-4')))}},
       'featured_models': {
          'openai': (
              model_configs.get_provider_model_config(('openai', 'gpt-4')),),
          'claude': (
              ('claude', 'opus-4'),)},
       'models_by_call_type': {
          types.CallType.GENERATE_TEXT: {
              'openai': (
                  model_configs.get_provider_model_config(('openai', 'gpt-4')),),
              'claude': (
                  ('claude', 'sonnet-4'),)}},
       'models_by_size': {
          types.ModelSizeType.SMALL: (
              ('openai', 'gpt-4o-mini'),),
          types.ModelSizeType.LARGE: (
              model_configs.get_provider_model_config(('openai', 'gpt-4')),
              ('claude', 'opus-4'),)},
       'default_model_priority_list': (
          model_configs.get_provider_model_config(('openai', 'gpt-4')),
          ('claude', 'opus-4'),
          ('openai', 'o3-mini'),)},]


def _get_model_configs_schema_type_options():
  return [
      {},
      {'metadata': types.ModelConfigsSchemaMetadataType(
          version='1.0.0')},
      {'metadata': types.ModelConfigsSchemaMetadataType(
          version='1.0.0',
          released_at=datetime.datetime.now(datetime.timezone.utc),
          config_origin=types.ConfigOriginType.BUILT_IN,
          release_notes='Initial release')},
      {'version_config': types.ModelConfigsSchemaVersionConfigType(
          provider_model_configs={
              'openai': {
                  'gpt-4': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('openai', 'gpt-4')))}})},
      {'version_config': types.ModelConfigsSchemaVersionConfigType(
          featured_models={
              'openai': (
                  model_configs.get_provider_model_config(('openai', 'gpt-4')),)})},
      {'metadata': types.ModelConfigsSchemaMetadataType(
          version='1.0.0',
          released_at=datetime.datetime.now(datetime.timezone.utc)),
       'version_config': types.ModelConfigsSchemaVersionConfigType(
          provider_model_configs={
              'openai': {
                  'gpt-4': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('openai', 'gpt-4')))}})},
      {'metadata': types.ModelConfigsSchemaMetadataType(
          version='2.1.0',
          released_at=datetime.datetime.now(datetime.timezone.utc),
          config_origin=types.ConfigOriginType.PROXDASH,
          release_notes='Added new models and updated pricing'),
       'version_config': types.ModelConfigsSchemaVersionConfigType(
          provider_model_configs={
              'openai': {
                  'gpt-4': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('openai', 'gpt-4')),
                      pricing=types.ProviderModelPricingType(
                          per_response_token_cost=0.001,
                          per_query_token_cost=0.002),
                      features=types.ProviderModelFeatureType(
                          not_supported_features=['feature1']),
                      metadata=types.ProviderModelMetadataType(
                          call_type=types.CallType.GENERATE_TEXT,
                          is_featured=True,
                          model_size=types.ModelSizeType.LARGE)),
                  'o3-mini': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('openai', 'o3-mini')))},
              'claude': {
                  'opus-4': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('claude', 'opus-4')),
                      metadata=types.ProviderModelMetadataType(
                          is_featured=True,
                          model_size=types.ModelSizeType.LARGEST)),
                  'sonnet-4': types.ProviderModelConfigType(
                      provider_model=model_configs.get_provider_model_config(
                          ('claude', 'sonnet-4')))}},
          featured_models={
              'openai': (
                  model_configs.get_provider_model_config(('openai', 'gpt-4')),),
              'claude': (
                  ('claude', 'opus-4'),)},
          models_by_call_type={
              types.CallType.GENERATE_TEXT: {
                  'openai': (
                      model_configs.get_provider_model_config(
                          ('openai', 'gpt-4')),),
                  'claude': (
                      ('claude', 'sonnet-4'),)}},
          models_by_size={
              types.ModelSizeType.SMALL: (
                  ('openai', 'gpt-4o-mini'),),
              types.ModelSizeType.LARGE: (
                  model_configs.get_provider_model_config(('openai', 'gpt-4')),
                  ('claude', 'opus-4'),)},
          default_model_priority_list=(
              model_configs.get_provider_model_config(('openai', 'gpt-4')),
              ('claude', 'opus-4'),
              ('openai', 'o3-mini'),))},]


def _get_query_record_options():
  return [
      {'call_type': types.CallType.GENERATE_TEXT},
      {'provider_model': model_configs.get_provider_model_config(('openai', 'gpt-4'))},
      {'prompt': 'Hello, world!'},
      {'system': 'Hello, system!'},
      {'messages': [{'role': 'user', 'content': 'Hello, user!'}]},
      {'max_tokens': 100},
      {'temperature': 0.5},
      {'stop': ['stop']},
      {'token_count': 100},
      {'call_type': types.CallType.GENERATE_TEXT,
       'provider_model': model_configs.get_provider_model_config(('openai', 'gpt-4')),
       'prompt': 'Hello, world!',
       'system': 'Hello, system!',
       'messages': [{'role': 'user', 'content': 'Hello, user!'}],
       'max_tokens': 100,
       'temperature': 0.5,
       'stop': ['stop'],
       'token_count': 100},]


def _get_query_response_record_options():
  return [
      {'response': 'Hello, world!'},
      {'error': 'Error message'},
      {'start_utc_date': datetime.datetime.now(datetime.timezone.utc)},
      {'end_utc_date': datetime.datetime.now(datetime.timezone.utc)},
      {'local_time_offset_minute': (
          datetime.datetime.now().astimezone().utcoffset().total_seconds()
          // 60) * -1},
      {'response_time': datetime.timedelta(seconds=1)},
      {'estimated_cost': 1},
      {'token_count': 100},
      {'response': 'Hello, world!',
       'error': 'Error message',
       'start_utc_date': datetime.datetime.now(datetime.timezone.utc),
       'end_utc_date': datetime.datetime.now(datetime.timezone.utc),
       'local_time_offset_minute': (
          datetime.datetime.now().astimezone().utcoffset().total_seconds()
          // 60) * -1,
       'response_time': datetime.timedelta(seconds=1),
       'estimated_cost': 1,
       'token_count': 100},]


def _get_cache_record_options():
  return [
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT)},
      {'query_responses': [types.QueryResponseRecord(
          response='Hello, world!')]},
      {'shard_id': 0},
      {'shard_id': 'backlog'},
      {'last_access_time': datetime.datetime.now()},
      {'call_count': 1},
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT),
       'query_responses': [types.QueryResponseRecord(
          response='Hello, world!')],
       'shard_id': 0,
       'last_access_time': datetime.datetime.now(),
       'call_count': 1},]


def _get_light_cache_record_options():
  return [
      {'query_record_hash': 'hash_value'},
      {'query_response_count': 1},
      {'shard_id': 0},
      {'last_access_time': datetime.datetime.now()},
      {'call_count': 1},
      {'query_record_hash': 'hash_value',
       'query_response_count': 1,
       'shard_id': 0,
       'last_access_time': datetime.datetime.now(),
       'call_count': 1},]


def _get_logging_record_options():
  return [
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT)},
      {'response_record': types.QueryResponseRecord(
          response='Hello, world!')},
      {'response_source': types.ResponseSource.CACHE},
      {'look_fail_reason': types.CacheLookFailReason.CACHE_NOT_FOUND},
      {'query_record': types.QueryRecord(
          call_type=types.CallType.GENERATE_TEXT),
       'response_record': types.QueryResponseRecord(
          response='Hello, world!'),
       'response_source': types.ResponseSource.CACHE,
       'look_fail_reason': types.CacheLookFailReason.CACHE_NOT_FOUND},]


def _get_logging_options_options():
  return [
      {},
      {'logging_path': 'logging_path'},
      {'stdout': True},
      {'hide_sensitive_content': True},]


def _get_cache_options_options():
  return [
      {},
      {'cache_path': 'cache_path'},
      {'unique_response_limit': 1},
      {'retry_if_error_cached': True},
      {'clear_query_cache_on_connect': True},
      {'clear_model_cache_on_connect': True},]


def _get_proxdash_options_options():
  return [
      {},
      {'stdout': True},
      {'hide_sensitive_content': True},
      {'disable_proxdash': True},]


def _get_run_options_options():
  return [
      {},
      {'run_type': types.RunType.TEST},
      {'hidden_run_key': 'hidden_run_key'},
      {'experiment_path': 'experiment_path'},
      {'root_logging_path': 'root_logging_path'},
      {'default_model_cache_path': 'default_model_cache_path'},
      {'logging_options': types.LoggingOptions(
          logging_path='logging_path',
          stdout=True,
          hide_sensitive_content=True)},
      {'cache_options': types.CacheOptions(
          cache_path='cache_path',
          unique_response_limit=1,
          retry_if_error_cached=True,
          clear_query_cache_on_connect=True,
          clear_model_cache_on_connect=True)},
      {'proxdash_options': types.ProxDashOptions(
          stdout=True,
          hide_sensitive_content=True,
          disable_proxdash=True)},
      {'allow_multiprocessing': True},
      {'model_test_timeout': 25},
      {'strict_feature_test': True},
      {'suppress_provider_errors': True},
      {'run_type': types.RunType.TEST,
       'hidden_run_key': 'hidden_run_key',
       'experiment_path': 'experiment_path',
       'root_logging_path': 'root_logging_path',
       'default_model_cache_path': 'default_model_cache_path',
       'logging_options': types.LoggingOptions(
          logging_path='logging_path',
          stdout=True,
          hide_sensitive_content=True),
       'cache_options': types.CacheOptions(
          cache_path='cache_path',
          unique_response_limit=1,
          retry_if_error_cached=True,
          clear_query_cache_on_connect=True,
          clear_model_cache_on_connect=True),
       'proxdash_options': types.ProxDashOptions(
          stdout=True,
          hide_sensitive_content=True,
          disable_proxdash=True),
       'allow_multiprocessing': True,
       'model_test_timeout': 25},]


def _get_model_status_options():
  model_1 = model_configs.get_provider_model_config(('openai', 'gpt-4'))
  model_2 = model_configs.get_provider_model_config(('openai', 'o3-mini'))
  model_3 = model_configs.get_provider_model_config(('claude', 'opus-4'))
  model_4 = model_configs.get_provider_model_config(('claude', 'sonnet-4'))
  return [
      {},
      {'unprocessed_models': {model_1}},
      {'working_models': {model_1, model_2}},
      {'failed_models': {model_1, model_2, model_3}},
      {'filtered_models': {model_1, model_2, model_3, model_4}},
      {'provider_queries': {
          model_1: types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_1),
              response_record=types.QueryResponseRecord(
                  response='Hello, world!'))}},
      {'unprocessed_models': {model_1},
       'working_models': {model_2},
       'failed_models': {model_3},
       'filtered_models': {model_4}},
       {'unprocessed_models': {model_1, model_2},
       'working_models': {model_2, model_3},
       'failed_models': {model_3, model_4},
       'filtered_models': {model_4, model_1},
       'provider_queries': {
          model_1: types.LoggingRecord(
              query_record=types.QueryRecord(
                  call_type=types.CallType.GENERATE_TEXT,
                  provider_model=model_1),
              response_record=types.QueryResponseRecord(
                  response='Hello, world!'))}}]


def _get_base_provider_stats_options():
  return [
    {'total_queries': 1},
    {'total_successes': 2},
    {'total_fails': 3},
    {'total_token_count': 4},
    {'total_query_token_count': 5},
    {'total_response_token_count': 6},
    {'total_successes': 1,
     'total_response_time': 7.0},
    {'estimated_cost': 8.0},
    {'total_cache_look_fail_reasons': {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 9}},
    {'total_queries': 1,
     'total_successes': 2,
     'total_fails': 3,
     'total_token_count': 4,
     'total_query_token_count': 5,
     'total_response_token_count': 6,
     'total_successes': 1,
     'total_response_time': 7.0,
     'estimated_cost': 8.0,
     'total_cache_look_fail_reasons': {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 9}}]


def _get_base_cache_stats_options():
  return [
    {'total_cache_hit': 1},
    {'total_success_return': 2},
    {'total_fail_return': 3},
    {'saved_token_count': 4},
    {'saved_query_token_count': 5},
    {'saved_response_token_count': 6},
    {'total_success_return': 1,
     'saved_total_response_time': 7.0},
    {'saved_estimated_cost': 8.0},
    {'total_cache_hit': 1,
     'total_success_return': 2,
     'total_fail_return': 3,
     'saved_token_count': 4,
     'saved_query_token_count': 5,
     'saved_response_token_count': 6,
     'saved_total_response_time': 7.0,
     'saved_estimated_cost': 8.0}]


def _get_provider_model_stats_options():
  return [
    {'provider_model': model_configs.get_provider_model_config(('openai', 'gpt-4'))},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'provider_model': model_configs.get_provider_model_config(('openai', 'gpt-4')),
     'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)}]


def _get_provider_stats_options():
  return [
    {'provider': 'openai'},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'provider_models': {
        model_configs.get_provider_model_config(
            ('openai', 'gpt-4')): stat_types.ProviderModelStats()}},
    {'provider': 'openai',
     'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1),
     'provider_models': {
        model_configs.get_provider_model_config(
            ('openai', 'gpt-4')): stat_types.ProviderModelStats()}}]


def _get_run_stats_options():
  return [
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1)},
    {'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)},
    {'provider_stats': stat_types.BaseProviderStats(total_queries=1),
     'cache_stats': stat_types.BaseCacheStats(total_cache_hit=1)}]


def _get_provider_model_configs_type_options():
  return [
    {'openai': {
        'gpt-4': types.ProviderModelConfigType(
            provider_model=model_configs.get_provider_model_config(
                ('openai', 'gpt-4')))}},
    {'openai': {
        'gpt-4': types.ProviderModelConfigType(
            provider_model=model_configs.get_provider_model_config(
                ('openai', 'gpt-4')),
            pricing=types.ProviderModelPricingType(
                per_response_token_cost=0.001,
                per_query_token_cost=0.002))},
     'claude': {
        'opus-4': types.ProviderModelConfigType(
            provider_model=model_configs.get_provider_model_config(
                ('claude', 'opus-4')),
            metadata=types.ProviderModelMetadataType(
                is_featured=True))}},]


def _get_featured_models_type_options():
  return [
    {'openai': (
        model_configs.get_provider_model_config(('openai', 'gpt-4')),)},
    {'openai': (
        model_configs.get_provider_model_config(('openai', 'gpt-4')),),
     'claude': (
        ('claude', 'opus-4'),)},
    {'openai': (
        model_configs.get_provider_model_config(('openai', 'gpt-4')),
        ('openai', 'o3-mini'),),
     'claude': (
        ('claude', 'opus-4'),
        model_configs.get_provider_model_config(('claude', 'sonnet-4')),)},]


def _get_models_by_call_type_type_options():
  return [
    {types.CallType.GENERATE_TEXT: {
        'openai': (
            model_configs.get_provider_model_config(('openai', 'gpt-4')),)}},
    {types.CallType.GENERATE_TEXT: {
        'openai': (
            model_configs.get_provider_model_config(('openai', 'gpt-4')),),
        'claude': (
            ('claude', 'sonnet-4'),)}},
    {types.CallType.GENERATE_TEXT: {
        'openai': (
            model_configs.get_provider_model_config(('openai', 'gpt-4')),
            ('openai', 'o3-mini'),),
        'claude': (
            ('claude', 'sonnet-4'),
            model_configs.get_provider_model_config(('claude', 'opus-4')),)}},]


def _get_models_by_size_type_options():
  return [
    {types.ModelSizeType.SMALL: (
        ('openai', 'gpt-4o-mini'),)},
    {types.ModelSizeType.SMALL: (
        ('openai', 'gpt-4o-mini'),),
     types.ModelSizeType.LARGE: (
        model_configs.get_provider_model_config(('openai', 'gpt-4')),)},
    {types.ModelSizeType.SMALL: (
        ('openai', 'gpt-4o-mini'),),
     types.ModelSizeType.LARGE: (
        model_configs.get_provider_model_config(('openai', 'gpt-4')),
        ('claude', 'opus-4'),)},]


def _get_default_model_priority_list_type_options():
  return [
    (model_configs.get_provider_model_config(('openai', 'gpt-4')),),
    (model_configs.get_provider_model_config(('openai', 'gpt-4')),
     ('claude', 'opus-4'),),
    (model_configs.get_provider_model_config(('openai', 'gpt-4')),
     ('claude', 'opus-4'),
     ('openai', 'o3-mini'),),]


class TestTypeSerializer:
  @pytest.mark.parametrize(
      'provider_model_type_options',
      _get_provider_model_type_options())
  def test_encode_decode_provider_model_type(self, provider_model_type_options):
    provider_model_type = model_configs.get_provider_model_config(
        (provider_model_type_options['provider'],
         provider_model_type_options['model']))
    encoded_provider_model_type = type_serializer.encode_provider_model_type(
        provider_model_type=provider_model_type)
    decoded_provider_model_type = type_serializer.decode_provider_model_type(
        record=encoded_provider_model_type)
    assert provider_model_type == decoded_provider_model_type

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_get_query_record_hash(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    query_hash_value = hash_serializer.get_query_record_hash(
        query_record=query_record)

    query_record_options['max_tokens'] = 222
    query_record_2 = types.QueryRecord(**query_record_options)
    query_hash_value_2 = hash_serializer.get_query_record_hash(
        query_record=query_record_2)

    assert query_hash_value != query_hash_value_2
    assert query_hash_value == (
        hash_serializer.get_query_record_hash(
            query_record=query_record))

  @pytest.mark.parametrize('query_record_options', _get_query_record_options())
  def test_encode_decode_query_record(self, query_record_options):
    query_record = types.QueryRecord(**query_record_options)
    encoded_query_record = type_serializer.encode_query_record(
        query_record=query_record)
    decoded_query_record = type_serializer.decode_query_record(
        record=encoded_query_record)
    assert query_record == decoded_query_record

  @pytest.mark.parametrize(
      'query_response_record_options', _get_query_response_record_options())
  def test_encode_decode_query_response_record(
      self, query_response_record_options):
    query_response_record = types.QueryResponseRecord(
        **query_response_record_options)
    encoded_query_response_record = (
        type_serializer.encode_query_response_record(
            query_response_record=query_response_record))
    decoded_query_response_record = (
        type_serializer.decode_query_response_record(
            record=encoded_query_response_record))
    assert query_response_record == decoded_query_response_record

  @pytest.mark.parametrize('cache_record_options', _get_cache_record_options())
  def test_encode_decode_cache_record(self, cache_record_options):
    cache_record = types.CacheRecord(**cache_record_options)
    encoded_cache_record = type_serializer.encode_cache_record(
        cache_record=cache_record)
    decoded_cache_record = type_serializer.decode_cache_record(
        record=encoded_cache_record)
    assert cache_record == decoded_cache_record

  @pytest.mark.parametrize(
      'light_cache_record_options', _get_light_cache_record_options())
  def test_encode_decode_light_cache_record(self, light_cache_record_options):
    light_cache_record = types.LightCacheRecord(**light_cache_record_options)
    encoded_light_cache_record = (
        type_serializer.encode_light_cache_record(
            light_cache_record=light_cache_record))
    decoded_light_cache_record = (
        type_serializer.decode_light_cache_record(
            record=encoded_light_cache_record))
    assert light_cache_record == decoded_light_cache_record

  @pytest.mark.parametrize(
      'logging_record_options', _get_logging_record_options())
  def test_encode_decode_logging_record(self, logging_record_options):
    logging_record = types.LoggingRecord(**logging_record_options)
    encoded_logging_record = type_serializer.encode_logging_record(
        logging_record=logging_record)
    decoded_logging_record = type_serializer.decode_logging_record(
        record=encoded_logging_record)
    assert logging_record == decoded_logging_record

  @pytest.mark.parametrize(
      'logging_options_options', _get_logging_options_options())
  def test_encode_decode_logging_options(self, logging_options_options):
    logging_options = types.LoggingOptions(**logging_options_options)
    encoded_logging_options = type_serializer.encode_logging_options(
        logging_options=logging_options)
    decoded_logging_options = type_serializer.decode_logging_options(
        record=encoded_logging_options)
    assert logging_options == decoded_logging_options

  @pytest.mark.parametrize(
      'cache_options_options', _get_cache_options_options())
  def test_encode_decode_cache_options(self, cache_options_options):
    cache_options = types.CacheOptions(**cache_options_options)
    encoded_cache_options = type_serializer.encode_cache_options(
        cache_options=cache_options)
    decoded_cache_options = type_serializer.decode_cache_options(
        record=encoded_cache_options)
    assert cache_options == decoded_cache_options

  @pytest.mark.parametrize(
      'proxdash_options_options', _get_proxdash_options_options())
  def test_encode_decode_proxdash_options(self, proxdash_options_options):
    proxdash_options = types.ProxDashOptions(**proxdash_options_options)
    encoded_proxdash_options = type_serializer.encode_proxdash_options(
        proxdash_options=proxdash_options)
    decoded_proxdash_options = type_serializer.decode_proxdash_options(
        record=encoded_proxdash_options)
    assert proxdash_options == decoded_proxdash_options

  @pytest.mark.parametrize(
      'run_options_options', _get_run_options_options())
  def test_encode_decode_run_options(self, run_options_options):
    run_options = types.RunOptions(**run_options_options)
    encoded_run_options = type_serializer.encode_run_options(
        run_options=run_options)
    decoded_run_options = type_serializer.decode_run_options(
        record=encoded_run_options)
    assert run_options == decoded_run_options

  @pytest.mark.parametrize('model_status_options', _get_model_status_options())
  def test_encode_decode_model_status(self, model_status_options):
    model_status = types.ModelStatus(**model_status_options)
    encoded_model_status = type_serializer.encode_model_status(
        model_status=model_status)
    decoded_model_status = type_serializer.decode_model_status(
        record=encoded_model_status)
    assert model_status == decoded_model_status

  @pytest.mark.parametrize(
      'base_provider_stats_options', _get_base_provider_stats_options())
  def test_encode_decode_base_provider_stats(
      self, base_provider_stats_options):
    provider_stats = stat_types.BaseProviderStats(**base_provider_stats_options)
    encoded_provider_stats = type_serializer.encode_base_provider_stats(
        base_provider_stats=provider_stats)
    decoded_provider_stats = type_serializer.decode_base_provider_stats(
        record=encoded_provider_stats)
    assert provider_stats == decoded_provider_stats

  @pytest.mark.parametrize(
      'base_cache_stats_options', _get_base_cache_stats_options())
  def test_encode_decode_base_cache_stats(self, base_cache_stats_options):
    cache_stats = stat_types.BaseCacheStats(**base_cache_stats_options)
    encoded_cache_stats = type_serializer.encode_base_cache_stats(
        base_cache_stats=cache_stats)
    decoded_cache_stats = type_serializer.decode_base_cache_stats(
        record=encoded_cache_stats)
    assert cache_stats == decoded_cache_stats

  @pytest.mark.parametrize(
      'provider_model_stats_options',
      _get_provider_model_stats_options())
  def test_encode_decode_provider_model_stats(
      self, provider_model_stats_options):
    provider_model_stats = stat_types.ProviderModelStats(
        **provider_model_stats_options)
    encoded_provider_model_stats = type_serializer.encode_provider_model_stats(
        provider_model_stats=provider_model_stats)
    decoded_provider_model_stats = type_serializer.decode_provider_model_stats(
        record=encoded_provider_model_stats)
    assert provider_model_stats == decoded_provider_model_stats

  @pytest.mark.parametrize(
      'provider_stats_options', _get_provider_stats_options())
  def test_encode_decode_provider_stats(self, provider_stats_options):
    provider_stats = stat_types.ProviderStats(**provider_stats_options)
    encoded_provider_stats = type_serializer.encode_provider_stats(
        provider_stats=provider_stats)
    decoded_provider_stats = type_serializer.decode_provider_stats(
        record=encoded_provider_stats)
    assert provider_stats == decoded_provider_stats

  @pytest.mark.parametrize('run_stats_options', _get_run_stats_options())
  def test_encode_decode_run_stats(self, run_stats_options):
    run_stats = stat_types.RunStats(**run_stats_options)
    encoded_run_stats = type_serializer.encode_run_stats(
        run_stats=run_stats)
    decoded_run_stats = type_serializer.decode_run_stats(
        record=encoded_run_stats)
    assert run_stats == decoded_run_stats

  @pytest.mark.parametrize(
      'provider_model_identifier',
      _get_provider_model_identifier_options())
  def test_encode_decode_provider_model_identifier(
      self, provider_model_identifier):
    encoded_provider_model_identifier = (
        type_serializer.encode_provider_model_identifier(
            provider_model_identifier=provider_model_identifier))
    decoded_provider_model_identifier = (
        type_serializer.decode_provider_model_identifier(
            record=encoded_provider_model_identifier))
    assert provider_model_identifier == decoded_provider_model_identifier

  @pytest.mark.parametrize(
      'provider_model_pricing_type_options',
      _get_provider_model_pricing_type_options())
  def test_encode_decode_provider_model_pricing_type(
      self, provider_model_pricing_type_options):
    provider_model_pricing_type = types.ProviderModelPricingType(
        **provider_model_pricing_type_options)
    encoded_provider_model_pricing_type = (
        type_serializer.encode_provider_model_pricing_type(
            provider_model_pricing_type=provider_model_pricing_type))
    decoded_provider_model_pricing_type = (
        type_serializer.decode_provider_model_pricing_type(
            record=encoded_provider_model_pricing_type))
    assert provider_model_pricing_type == decoded_provider_model_pricing_type

  @pytest.mark.parametrize(
      'provider_model_feature_type_options',
      _get_provider_model_feature_type_options())
  def test_encode_decode_provider_model_feature_type(
      self, provider_model_feature_type_options):
    provider_model_feature_type = types.ProviderModelFeatureType(
        **provider_model_feature_type_options)
    encoded_provider_model_feature_type = (
        type_serializer.encode_provider_model_feature_type(
            provider_model_feature_type=provider_model_feature_type))
    decoded_provider_model_feature_type = (
        type_serializer.decode_provider_model_feature_type(
            record=encoded_provider_model_feature_type))
    assert provider_model_feature_type == decoded_provider_model_feature_type

  @pytest.mark.parametrize(
      'provider_model_metadata_type_options',
      _get_provider_model_metadata_type_options())
  def test_encode_decode_provider_model_metadata_type(
      self, provider_model_metadata_type_options):
    provider_model_metadata_type = types.ProviderModelMetadataType(
        **provider_model_metadata_type_options)
    encoded_provider_model_metadata_type = (
        type_serializer.encode_provider_model_metadata_type(
            provider_model_metadata_type=provider_model_metadata_type))
    decoded_provider_model_metadata_type = (
        type_serializer.decode_provider_model_metadata_type(
            record=encoded_provider_model_metadata_type))
    assert provider_model_metadata_type == decoded_provider_model_metadata_type

  @pytest.mark.parametrize(
      'provider_model_config_type_options',
      _get_provider_model_config_type_options())
  def test_encode_decode_provider_model_config_type(
      self, provider_model_config_type_options):
    provider_model_config_type = types.ProviderModelConfigType(
        **provider_model_config_type_options)
    encoded_provider_model_config_type = (
        type_serializer.encode_provider_model_config_type(
            provider_model_config_type=provider_model_config_type))
    decoded_provider_model_config_type = (
        type_serializer.decode_provider_model_config_type(
            record=encoded_provider_model_config_type))
    assert provider_model_config_type == decoded_provider_model_config_type

  @pytest.mark.parametrize(
      'model_configs_schema_metadata_type_options',
      _get_model_configs_schema_metadata_type_options())
  def test_encode_decode_model_configs_schema_metadata_type(
      self, model_configs_schema_metadata_type_options):
    model_configs_schema_metadata_type = types.ModelConfigsSchemaMetadataType(
        **model_configs_schema_metadata_type_options)
    encoded_model_configs_schema_metadata_type = (
        type_serializer.encode_model_configs_schema_metadata_type(
            model_configs_schema_metadata_type=(
                model_configs_schema_metadata_type)))
    decoded_model_configs_schema_metadata_type = (
        type_serializer.decode_model_configs_schema_metadata_type(
            record=encoded_model_configs_schema_metadata_type))
    assert model_configs_schema_metadata_type == (
        decoded_model_configs_schema_metadata_type)

  @pytest.mark.parametrize(
      'model_configs_schema_version_config_type_options',
      _get_model_configs_schema_version_config_type_options())
  def test_encode_decode_model_configs_schema_version_config_type(
      self, model_configs_schema_version_config_type_options):
    model_configs_schema_version_config_type = (
        types.ModelConfigsSchemaVersionConfigType(
            **model_configs_schema_version_config_type_options))
    encoded_model_configs_schema_version_config_type = (
        type_serializer.encode_model_configs_schema_version_config_type(
            model_configs_schema_version_config_type=(
                model_configs_schema_version_config_type)))
    decoded_model_configs_schema_version_config_type = (
        type_serializer.decode_model_configs_schema_version_config_type(
            record=encoded_model_configs_schema_version_config_type))
    assert model_configs_schema_version_config_type == (
        decoded_model_configs_schema_version_config_type)

  @pytest.mark.parametrize(
      'model_configs_schema_type_options',
      _get_model_configs_schema_type_options())
  def test_encode_decode_model_configs_schema_type(
      self, model_configs_schema_type_options):
    model_configs_schema_type = types.ModelConfigsSchemaType(
        **model_configs_schema_type_options)
    encoded_model_configs_schema_type = (
        type_serializer.encode_model_configs_schema_type(
            model_configs_schema_type=model_configs_schema_type))
    decoded_model_configs_schema_type = (
        type_serializer.decode_model_configs_schema_type(
            record=encoded_model_configs_schema_type))
    assert model_configs_schema_type == decoded_model_configs_schema_type

  @pytest.mark.parametrize(
      'provider_model_configs_type_options',
      _get_provider_model_configs_type_options())
  def test_encode_decode_provider_model_configs_type(
      self, provider_model_configs_type_options):
    encoded_provider_model_configs_type = (
        type_serializer.encode_provider_model_configs_type(
            provider_model_configs=provider_model_configs_type_options))
    decoded_provider_model_configs_type = (
        type_serializer.decode_provider_model_configs_type(
            record=encoded_provider_model_configs_type))
    assert provider_model_configs_type_options == (
        decoded_provider_model_configs_type)

  @pytest.mark.parametrize(
      'featured_models_type_options',
      _get_featured_models_type_options())
  def test_encode_decode_featured_models_type(
      self, featured_models_type_options):
    encoded_featured_models_type = (
        type_serializer.encode_featured_models_type(
            featured_models=featured_models_type_options))
    decoded_featured_models_type = (
        type_serializer.decode_featured_models_type(
            record=encoded_featured_models_type))
    assert featured_models_type_options == decoded_featured_models_type

  @pytest.mark.parametrize(
      'models_by_call_type_type_options',
      _get_models_by_call_type_type_options())
  def test_encode_decode_models_by_call_type_type(
      self, models_by_call_type_type_options):
    encoded_models_by_call_type_type = (
        type_serializer.encode_models_by_call_type_type(
            models_by_call_type=models_by_call_type_type_options))
    decoded_models_by_call_type_type = (
        type_serializer.decode_models_by_call_type_type(
            record=encoded_models_by_call_type_type))
    assert models_by_call_type_type_options == decoded_models_by_call_type_type

  @pytest.mark.parametrize(
      'models_by_size_type_options',
      _get_models_by_size_type_options())
  def test_encode_decode_models_by_size_type(
      self, models_by_size_type_options):
    encoded_models_by_size_type = (
        type_serializer.encode_models_by_size_type(
            models_by_size=models_by_size_type_options))
    decoded_models_by_size_type = (
        type_serializer.decode_models_by_size_type(
            record=encoded_models_by_size_type))
    assert models_by_size_type_options == decoded_models_by_size_type

  @pytest.mark.parametrize(
      'default_model_priority_list_type_options',
      _get_default_model_priority_list_type_options())
  def test_encode_decode_default_model_priority_list_type(
      self, default_model_priority_list_type_options):
    encoded_default_model_priority_list_type = (
        type_serializer.encode_default_model_priority_list_type(
            default_model_priority_list=(
                default_model_priority_list_type_options)))
    decoded_default_model_priority_list_type = (
        type_serializer.decode_default_model_priority_list_type(
            record=encoded_default_model_priority_list_type))
    assert default_model_priority_list_type_options == (
        decoded_default_model_priority_list_type)

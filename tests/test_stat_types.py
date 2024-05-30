import copy
import proxai.types as types
import proxai.stat_types as stat_types
import pytest


def base_provider_stats_examples():
  example = stat_types.BaseProviderStats(
      total_queries=2,
      total_successes=1,
      total_fails=1,
      total_token_count=200,
      total_query_token_count=100,
      total_response_token_count=100,
      total_response_time=0.5,
      estimated_price=1.0,
      total_cache_look_fail_reasons={
          types.CacheLookFailReason.CACHE_NOT_FOUND: 1})
  big_example = stat_types.BaseProviderStats(
      total_queries=4,
      total_successes=2,
      total_fails=2,
      total_token_count=400,
      total_query_token_count=200,
      total_response_token_count=200,
      total_response_time=1.0,
      estimated_price=2.0,
      total_cache_look_fail_reasons={
          types.CacheLookFailReason.CACHE_NOT_FOUND: 2})
  return (example, big_example)


def base_cache_stats_examples():
  example = stat_types.BaseCacheStats(
      total_cache_hit=2,
      total_success_return=1,
      total_fail_return=1,
      saved_token_count=200,
      saved_query_token_count=100,
      saved_response_token_count=100,
      saved_total_response_time=0.5,
      saved_estimated_price=1.0)
  big_example = stat_types.BaseCacheStats(
      total_cache_hit=4,
      total_success_return=2,
      total_fail_return=2,
      saved_token_count=400,
      saved_query_token_count=200,
      saved_response_token_count=200,
      saved_total_response_time=1.0,
      saved_estimated_price=2.0)
  return (example, big_example)


def model_stats_examples(model=('openai', 'gpt-4')):
  base_provider_stats, total_base_provider_stats = base_provider_stats_examples()
  base_cache_stats, total_base_cache_stats = base_cache_stats_examples()
  example = stat_types.ModelStats(model=model)
  big_example = stat_types.ModelStats(model=model)
  example += base_provider_stats
  example += base_cache_stats
  big_example += total_base_provider_stats
  big_example += total_base_cache_stats
  return (example, big_example)


def provider_stats_examples(
    provider='openai',
    models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')]):
  example = stat_types.ProviderStats(provider=provider)
  big_example = stat_types.ProviderStats(provider=provider)
  for model in models:
    model_stats, total_model_stats = model_stats_examples(model)
    example += model_stats
    big_example += total_model_stats
  return (example, big_example)


def run_stats_examples(providers):
  example = stat_types.RunStats()
  big_example = stat_types.RunStats()
  for provider, models in providers.items():
    provider_example, total_provider_example = provider_stats_examples(
        provider=provider, models=models)
    example += provider_example
    big_example += total_provider_example
  return (example, big_example)


class TestStatTypes:
  def test_base_provider_stats_eq(self):
    example, _ = base_provider_stats_examples()
    example_2 = copy.deepcopy(example)
    assert example == example_2
    example_2.total_response_time = 1.0
    assert example != example_2

  def test_base_provider_stats_add_1(self):
    example, big_example = base_provider_stats_examples()
    assert example + example == big_example

  def test_base_provider_stats_add_2(self):
    example, big_example = base_provider_stats_examples()
    example_2 = copy.deepcopy(example)
    example_2.total_cache_look_fail_reasons = {
        types.CacheLookFailReason.CACHE_NOT_MATCHED: 1}
    big_example.total_cache_look_fail_reasons = {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 1,
        types.CacheLookFailReason.CACHE_NOT_MATCHED: 1}
    assert example + example_2 == big_example

  def test_base_provider_stats_sub_1(self):
    example, big_example = base_provider_stats_examples()
    assert big_example - example == example

  def test_base_provider_stats_sub_2(self):
    example, big_example = base_provider_stats_examples()
    example_2 = copy.deepcopy(example)
    example_2.total_cache_look_fail_reasons = {
        types.CacheLookFailReason.CACHE_NOT_MATCHED: 1}
    big_example.total_cache_look_fail_reasons = {
        types.CacheLookFailReason.CACHE_NOT_FOUND: 1,
        types.CacheLookFailReason.CACHE_NOT_MATCHED: 1}
    assert big_example - example_2 == example

  def test_invalid_base_provider_stats_sub(self):
    example, big_example = base_provider_stats_examples()
    with pytest.raises(ValueError):
      example - big_example

  def test_base_cache_stats_add(self):
    example, big_example = base_cache_stats_examples()
    assert example + example == big_example

  def test_base_cache_stats_sub(self):
    example, big_example = base_cache_stats_examples()
    assert big_example - example == example

  def test_invalid_base_cache_stats_sub(self):
    example, big_example = base_cache_stats_examples()
    with pytest.raises(ValueError):
      example - big_example

  def test_model_stats_add(self):
    example, big_example = model_stats_examples()
    assert example + example == big_example

  def test_invalid_model_stats_add(self):
    example, _ = model_stats_examples(model=('openai', 'gpt-3.5-turbo'))
    example_2, _ = model_stats_examples(model=('openai', 'gpt-4'))
    with pytest.raises(ValueError):
      example + example_2

  def test_model_stats_sub(self):
    example, big_example = model_stats_examples()
    assert big_example - example == example

  def test_invalid_model_stats_sub_1(self):
    example, big_example = model_stats_examples()
    with pytest.raises(ValueError):
      example - big_example

  def test_invalid_model_stats_sub_2(self):
    example, _ = model_stats_examples(model=('openai', 'gpt-3.5-turbo'))
    example_2, _ = model_stats_examples(model=('openai', 'gpt-4'))
    with pytest.raises(ValueError):
      example - example_2

  def test_provider_stats_add_1(self):
    example_1, example_2 = provider_stats_examples(
        provider='openai', models=[('openai', 'gpt-3.5-turbo')])
    example_3, example_4 = provider_stats_examples(
        provider='openai', models=[('openai', 'gpt-4')])
    total_example_1, total_example_2 = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    assert example_1 + example_3 == total_example_1
    assert example_2 + example_4 == total_example_2

  def test_provider_stats_add_2(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    example_2, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    _, big_example = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    assert example_1 + example_2 == big_example

  def test_provider_stats_add_model_stats(self):
    provider_example, big_provider_example = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo')])
    model_example, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    assert provider_example + model_example == big_provider_example

  def test_invalid_provider_stats_add(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    with pytest.raises(ValueError):
      example_1 + example_2

  def test_provider_stats_sub_1(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    assert example_1 - example_1 == stat_types.ProviderStats(
        provider='openai',
        provider_stats=stat_types.BaseProviderStats(),
        cache_stats=stat_types.BaseCacheStats(),
        models={})

  def test_provider_stats_sub_2(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    example_2, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    _, big_example = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    assert big_example - example_2 == example_1

  def test_provider_stats_sub_3(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo')])
    example_2, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-4')])
    big_example, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    assert big_example - example_2 == example_1

  def test_provider_stats_sub_model_stats(self):
    provider_example, big_provider_example = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo')])
    model_example, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    assert big_provider_example - model_example == provider_example

  def test_invalid_provider_stats_sub_1(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo')])
    example_2, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-4')])
    with pytest.raises(ValueError):
      example_1 - example_2

  def test_invalid_provider_stats_sub_2(self):
    example, big_example = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    with pytest.raises(ValueError):
      example - big_example

  def test_invalid_provider_stats_sub_3(self):
    example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo')])
    example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    with pytest.raises(ValueError):
      example_1 - example_2

  def test_run_stats_add_1(self):
    example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    _, big_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert example_1 + example_2 == big_example

  def test_run_stats_add_2(self):
    example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-4')]})
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert example_1 + example_2 == total_example

  def test_run_stats_add_3(self):
    run_stats_example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')]})
    run_stats_example_2, _ = run_stats_examples({
        'claude': [('claude', 'claude-3-opus-20240229')]})
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        stat_types.RunStats()
        + run_stats_example_1
        + run_stats_example_2) == total_example

  def test_run_stats_add_4(self):
    run_stats_example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    run_stats_example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')]})
    run_stats_example_3, _ = run_stats_examples({
        'claude': [('claude', 'claude-3-opus-20240229')]})
    _, big_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (run_stats_example_1
            + run_stats_example_2
            + run_stats_example_3) == big_example

  def test_run_stats_add_provider_stats_1(self):
    provider_stats_example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    provider_stats_example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        stat_types.RunStats()
        + provider_stats_example_1
        + provider_stats_example_2) == total_example

  def test_run_stats_add_provider_stats_2(self):
    run_stats_example, big_run_stats_example = run_stats_examples(
        {'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
         'claude': [('claude', 'claude-3-opus-20240229')]})
    provider_stats_example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    provider_stats_example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    assert (
        run_stats_example
        + provider_stats_example_1
        + provider_stats_example_2) == big_run_stats_example

  def test_run_stats_add_model_stats_1(self):
    model_stats_example_1, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    model_stats_example_2, _ = model_stats_examples(
        model=('claude', 'claude-3-opus-20240229'))
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        stat_types.RunStats()
        + model_stats_example_1
        + model_stats_example_2) == total_example

  def test_run_stats_add_model_stats_2(self):
    model_stats_example_1, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    model_stats_example_2, _ = model_stats_examples(
        model=('claude', 'claude-3-opus-20240229'))
    run_stats_example, big_run_stats_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        run_stats_example
        + model_stats_example_1
        + model_stats_example_2) == big_run_stats_example

  def test_run_stats_sub_1(self):
    example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    _, big_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert big_example - example_2 == example_1

  def test_run_stats_sub_2(self):
    example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-4')]})
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert total_example - example_2 == example_1

  def test_run_stats_sub_3(self):
    run_stats_example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')]})
    run_stats_example_2, _ = run_stats_examples({
        'claude': [('claude', 'claude-3-opus-20240229')]})
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (total_example
            - run_stats_example_2
            - run_stats_example_1) == stat_types.RunStats(
                provider_stats=stat_types.BaseProviderStats(),
                cache_stats=stat_types.BaseCacheStats())

  def test_run_stats_sub_4(self):
    run_stats_example_1, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    run_stats_example_2, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')]})
    run_stats_example_3, _ = run_stats_examples({
        'claude': [('claude', 'claude-3-opus-20240229')]})
    _, big_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (big_example
            - run_stats_example_3
            - run_stats_example_2) == run_stats_example_1

  def test_run_stats_sub_provider_stats_1(self):
    provider_stats_example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    provider_stats_example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        total_example
        - provider_stats_example_2
        - provider_stats_example_1) == stat_types.RunStats(
            provider_stats=stat_types.BaseProviderStats(),
            cache_stats=stat_types.BaseCacheStats())

  def test_run_stats_sub_provider_stats_2(self):
    run_stats_example, big_run_stats_example = run_stats_examples(
        {'openai': [('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')],
         'claude': [('claude', 'claude-3-opus-20240229')]})
    provider_stats_example_1, _ = provider_stats_examples(
        provider='openai',
        models=[('openai', 'gpt-3.5-turbo'), ('openai', 'gpt-4')])
    provider_stats_example_2, _ = provider_stats_examples(
        provider='claude',
        models=[('claude', 'claude-3-opus-20240229')])
    assert (
        big_run_stats_example
        - provider_stats_example_2
        - provider_stats_example_1) == run_stats_example

  def test_run_stats_sub_model_stats_1(self):
    model_stats_example_1, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    model_stats_example_2, _ = model_stats_examples(
        model=('claude', 'claude-3-opus-20240229'))
    total_example, _ = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        total_example
        - model_stats_example_2
        - model_stats_example_1) == stat_types.RunStats(
            provider_stats=stat_types.BaseProviderStats(),
            cache_stats=stat_types.BaseCacheStats())

  def test_run_stats_sub_model_stats_2(self):
    model_stats_example_1, _ = model_stats_examples(
        model=('openai', 'gpt-3.5-turbo'))
    model_stats_example_2, _ = model_stats_examples(
        model=('claude', 'claude-3-opus-20240229'))
    run_stats_example, big_run_stats_example = run_stats_examples({
        'openai': [('openai', 'gpt-3.5-turbo')],
        'claude': [('claude', 'claude-3-opus-20240229')]})
    assert (
        big_run_stats_example
        - model_stats_example_2
        - model_stats_example_1) == run_stats_example

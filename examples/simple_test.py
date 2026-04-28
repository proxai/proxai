import random
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import proxai as px

_TEST_API_KEY = '0kf9czc-mohonse9-mceo43ae1dd'

_LONG_TECHNICAL_PROMPT = """\
We are designing a production-grade, multi-tenant LLM proxy gateway
codenamed "Hermes" that will sit in front of approximately twenty upstream
model providers (OpenAI, Anthropic, Google Gemini, Mistral, Cohere,
DeepSeek, Grok, HuggingFace, Databricks, Azure OpenAI, AWS Bedrock, Vertex,
Together, Replicate, OpenRouter, Fireworks, Groq, Perplexity, Voyage, and
an internal on-prem fine-tuned Llama cluster). The gateway is the single
ingress point for every internal service in a 4000-engineer company that
needs LLM access.

Operational scale and SLOs:
- ~30k requests per minute at steady state, with batch-eval bursts up to
  120k QPM lasting 5-15 minutes, three to five times per day.
- p50 gateway overhead must stay under 12ms; p99 under 250ms; p999 under
  1.2s.
- Streaming endpoints must deliver the first SSE event within 80ms of the
  upstream sending its first token.
- Tenant fairness: no single tenant can consume more than 25% of any
  provider quota during contention, even if their nominal share is higher.
- Availability: 99.95% monthly for the gateway itself, measured per
  region.
- Data residency: EU tenants must never have their prompts traverse
  non-EU provider endpoints, even during failover.

We have already settled on the following building blocks and you do not
need to re-litigate them:
- Rust (axum + tokio) for the data plane.
- Postgres + Redis for control plane and ephemeral state.
- Envoy in front for TLS termination and mTLS to upstreams.
- Kafka for async usage/billing events.
- OpenTelemetry for traces, Prometheus for metrics, Loki for logs.

Open design questions where we want a thorough analysis:

1. Routing and fallback policy
   We want to express, per virtual model and per tenant, a routing policy
   that combines: (a) primary provider preference, (b) ordered fallback
   list with per-step circuit breaker thresholds, (c) per-feature
   compatibility filters (e.g. only providers that support 32k context
   AND tool-calling AND JSON mode), (d) cost-aware degradation (allow
   falling back to a cheaper provider if upstream queue depth exceeds N),
   and (e) sticky session affinity for multi-turn chats so the same
   physical model serves the whole conversation.
   - How would you represent this policy declaratively? Consider whether
     a CEL/Rego DSL, a typed config struct, or a small interpreted
     pipeline gives the best balance of expressiveness and reviewability.
   - Where should the policy be evaluated: at the edge (Envoy filter), in
     the gateway's request handler, or in a dedicated "router" sidecar?
     Justify in terms of latency cost and blast radius of bad config.
   - How do we hot-reload policies without dropping in-flight streaming
     connections, and how do we A/B test a new routing rule on 1% of
     traffic per tenant without polluting the cache?

2. Caching strategy
   We expect ~35% of prompts to be exact duplicates within a 30-minute
   window (RAG with stable retrievers, agent loops, eval harnesses). We
   want a cache that:
   - Is keyed by a stable hash of (tenant, virtual model, full message
     array, all sampling parameters that affect output, system prompt,
     tool schema, response_format).
   - Respects tenant isolation cryptographically, not just by namespace.
   - Survives gateway pod restarts but does not sit in the hot path
     under Redis when the local in-process LRU is warm.
   - Handles streaming responses correctly: a streamed response must be
     cacheable and replayable as a stream, with the same chunk boundaries
     and inter-token timing distribution as the original (or better).
   - Has a per-tenant TTL override and supports a "no-cache" header for
     correctness-sensitive callers.
   Walk through your hash function, your cache hierarchy (L1 in-process,
   L2 regional Redis, L3 cross-region object store?), and your eviction
   policy. Quantify the memory budget you would target on a 64GB pod.

3. Rate limiting and quota
   Each provider gives us a global TPM (tokens per minute) and RPM
   (requests per minute) limit per API key, sometimes split by model
   family. Some providers (Anthropic, OpenAI) also impose a hidden burst
   limit that isn't documented. Tenants have their own per-tenant quotas,
   billed monthly, expressed in dollars rather than tokens. We need:
   - A distributed token-bucket implementation that is correct under
     network partitions, not just "eventually consistent". What is your
     stance on lazy vs. eager refill, and how do you avoid the thundering
     herd when a bucket refills?
   - A way to *estimate* token cost of a request before issuing it, so we
     can reject early. How accurate does this need to be, and what do you
     do when the estimate is wrong by more than 20%?
   - A graceful degradation story when a tenant approaches their dollar
     budget: do you switch them to a cheaper model, queue, hard-reject,
     or return a partial response with a warning?

4. Observability and cost attribution
   Every request must produce: a trace span with the full provider call
   chain, a structured log line, a usage event with token counts and
   estimated cost in nano-USD (so we avoid floating-point drift in
   billing), and a Prometheus metric. We also need:
   - Per-tenant, per-model, per-feature dashboards at 10-second
     granularity over a 30-day window.
   - Alerting on unusual cost spikes (sudden 5x increase in a tenant's
     hourly spend) without false positives during legitimate batch jobs.
   - Sampling: full traces for 1% of traffic, plus 100% of any request
     that hit an error or fell back beyond the second tier.
   How do you avoid the cardinality explosion on Prometheus when we have
   ~80 tenants x 200 models x 8 features?

5. Streaming correctness and backpressure
   SSE streams are the hardest part. Walk through:
   - How you propagate cancellation: when the client disconnects, the
     upstream call must be cancelled within 100ms, otherwise we pay for
     tokens nobody will read.
   - How you handle a slow client: if the consumer reads at 2 tok/s but
     the upstream produces at 80 tok/s, where does the buffer live?
     Concretely, what is your bounded buffer size and what happens when
     it fills?
   - How you reconnect a stream after a transient gateway failover.
     OpenAI does not support resume, so do you replay from cache,
     regenerate from the last seen offset, or surface the failure?
   - How you handle tool-call deltas, function-call JSON streaming, and
     reasoning-token streams (Anthropic's `thinking`, OpenAI's `o1`
     reasoning summaries) given that each provider serializes them
     differently.

6. Migration plan
   80 product teams currently use 20 different SDKs (anthropic-python,
   openai-python, vertexai, ...). We cannot ask them all to rewrite. We
   have committed to:
   - Phase 1: a drop-in replacement endpoint that mimics OpenAI's REST
     API so the openai-python SDK works unmodified by setting base_url.
   - Phase 2: a small "provider shim" library that lets teams keep their
     existing provider SDK but routes through Hermes for credential and
     metering purposes only.
   - Phase 3: a native Hermes SDK with first-class streaming, retry, and
     cache primitives.
   What goes wrong when you try to make Phase 1 OpenAI-compatible for
   non-OpenAI providers? Specifically, how do you map: Anthropic's
   `system` parameter, Anthropic's structured `tool_use` content blocks,
   Gemini's `parts` array, Mistral's `safe_prompt`, and Cohere's
   `connectors` field, into the OpenAI request shape without losing
   information? Where do you take the loss when the mapping is lossy?

7. Failure modes you would expect us to encounter and how you would
   defend against them:
   - A provider silently changes its tokenizer mid-quarter, breaking our
     pre-call cost estimate.
   - A tenant accidentally puts a 5MB system prompt in every request.
   - The Redis cluster fails over and the L2 cache is empty for 4
     minutes.
   - A new model version returns longer responses on average, blowing
     the token budget projections.
   - A misbehaving agent loop sends 50k requests in 60 seconds with the
     same tenant key.
   - A regional outage in us-east-1 forces 100% failover to eu-west-1,
     which violates EU residency for non-EU tenants.

Please answer in the form of a written design review. Structure your
response as: (a) a 5-bullet executive summary, (b) one section per
numbered topic above with concrete recommendations and the tradeoffs you
considered, (c) a list of decisions you would defer and why, (d) the
specific telemetry and SLOs you would put in place to know whether the
design is working in production. Cite specific numbers wherever you can,
and call out any place where you are guessing rather than citing
established practice. Finally, identify the three highest-risk parts of
this design and what you would prototype first to retire that risk.
"""


def simple_model_test():
  result = px.generate(
      'When is the first galatasaray and fenerbahce?')
  pprint(result)


def text_cache_test():
  client = px.Client(
      cache_options=px.CacheOptions(
          cache_path=f'{Path.home()}/proxai_cache/'),
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          # base_url='http://localhost:3001',
          api_key='hbk83g1-mohrw4rl-37x007op9r2',
      ))

  # client = px.Client()

  for i in range(100):
    result = client.generate(
        _LONG_TECHNICAL_PROMPT,
        # provider_model=('openai', 'gpt-5.5-pro')
        provider_model=('gemini', 'gemini-3-flash')
    )
    print(i, result.connection.result_source)


def image_cache_test():
  client = px.Client(
      logging_options=px.LoggingOptions(
          stdout=True,
      ),
      cache_options=px.CacheOptions(
          cache_path=f'{Path.home()}/proxai_cache/'))

  pprint(client.models.list_models(output_format='image'))

  result = client.generate_image(
      'Make funny cartoon cat in living room.',
      provider_model=('gemini', 'gemini-2.5-flash-image'))
  print(result.data[:100])


def list_models():
  # client = px.Client(
  #     provider_call_options=px.ProviderCallOptions(
  #         feature_mapping_strategy=px.FeatureMappingStrategy.STRICT))
  # models = client.models.list_provider_models(provider='gemini', feature_tags=['thinking'])
  # print(len(models))
  # for model in models:
  #   print(model)

  # client = px.Client(
  #     provider_call_options=px.ProviderCallOptions(
  #         feature_mapping_strategy=px.FeatureMappingStrategy.STRICT))
  # client = px.Client()
  # models = client.models.list_models(output_format='audio')
  # print(len(models))
  # for model in models:
  #   print(model)

  for provider in px.models.list_providers():
    print(f'#### {provider} ####')
    for size in ['small', 'medium', 'large', 'largest']:
      models = px.models.list_provider_models(provider, model_size=size)
      print(f'  === {size.upper()} ({len(models)}) ===')
      for model in models:
        model_config = px.models.get_model_config(model.provider, model.model)
        # pprint([x for x in model_config.metadata.tags if x.startswith('display::')])
        if "display::show_model=True" in model_config.metadata.tags:
          print(f'    * {model}')
        else:
          print(f'    {model}')

    # all_models = px.models.list_provider_models(provider)
    # untagged = [
    #     m for m in all_models
    #     if not px.models.get_model_config(
    #         m.provider, m.model).metadata.model_size_tags
    # ]
    # print(f'  === NO SIZE TAGS ({len(untagged)}) ===')
    # for model in untagged:
    #   print(f'    {model}')
    print()


def check_health():
  # client = px.Client(
  #     proxdash_options=px.ProxDashOptions(
  #         stdout=True,
  #         base_url='http://localhost:3001',
  #         api_key=_TEST_API_KEY,
  #     ),
  # )
  client = px.Client(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          api_key='hbk83g1-mohrw4rl-37x007op9r2',
      ),
  )
  client.models.check_health(verbose=True)


def proxdash_test():
  client = px.Client(
      proxdash_options=px.ProxDashOptions(
          stdout=True,
          base_url='http://localhost:3001',
          api_key=_TEST_API_KEY,
      ),
  )
  client.models.list_models()


def main():
  # simple_model_test()
  text_cache_test()
  # image_cache_test()
  # list_models()
  # check_health()
  # proxdash_test()

if __name__ == '__main__':
  main()

# Changelog

<!--next-version-placeholder-->

## v0.2.1 (2025/05/21)

- Added official website and GitHub repository links to package metadata

## v0.2.0 (2025/05/21)

- Version 0.2.0 release
- Key Features
  - Unified API
    - Cross-Provider Consistency: Interact with any supported AI model using the same API
    - Cross-Model Features: Best effort support for all text generation features for all models (message history, system prompt, temperature, max tokens, stop)
    - Strict Feature Control: Fail if the model does not support the feature
  - Provider Integrations
    - Major Providers: Connectors for Gemini, OpenAI, Claude, Grok, DeepSeek, Mistral, Cohere, Databricks, and HuggingFace
    - List Providers: Automatically detect available providers on your session
    - List Models: Automatically detect available models for each provider
    - Model Size Control: Get all models of a certain size: "small", "medium", "large", "largest" (of each provider)
    - Add More Providers: Add or update providers without code changes simply by adding API keys
  - Check Health
    - Check Health: Easy health summary of all providers and models
    - ProxDash: Health reports on ProxDash website to debug issues
  - Error Handling
    - Error Tracking: Robust error tracking with details on query records.
    - Error Control: Option of skipping errors or stopping
    - ProxDash: Tracking errors on ProxDash website
  - Caching System
    - Query Caching: Caching query results to reduce API calls and improve response time
    - Model Caching: Caching list of available models to speed up experiments
    - Extensive Configuration: Configure caching behavior with options like unique response limit, cache size, cache expiration time, etc.
    - ProxDash: Keep track of cache hits/misses, time saved, money saved, and more
  - Cost Estimation
    - Cost Estimation: Track estimated costs across different providers and models
    - ProxDash: Keep track of cost estimates: daily, weekly, monthly, per provider, per model, etc.
  - Logging System
    - Logging System: Detailed logging for debugging and monitoring
    - Hide Sensitive Information: Mask sensitive information like prompts, responses, and etc.
    - ProxDash: Keep track of logging information with different privacy settings


## v0.1.0 (4/04/2024)

- First release of `proxai`!

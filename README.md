# ProxAI

<div align="center">

<a href="https://proxai.co">
<img src="assets/proxai.png" alt="ProxAI Logo" width="200" style="border-radius: 10px;"/>
</a>

[![PyPI version](https://img.shields.io/badge/pip-v0.2.0-blue.svg)](https://pypi.org/project/proxai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/proxai/proxai/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1371968537446318191?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://discord.gg/Hxg6tPpX)

⚡️ **One Interface for Every AI Model!** ⚡️

[Official Site](https://proxai.co/) •
[Overview](https://www.proxai.co/overview) •
[Docs](https://proxai.co/proxai-docs) •
[Community](https://www.proxai.co/resources/community)

</div>

## Philosophy

ProxAI simplifies AI integration by providing a unified interface for connecting to multiple AI providers.

```python
    import proxai as px

    result = px.generate_text(
        'Explain what is ProxAI and why it is useful.',
        provider_model=('claude', 'sonnet'))
    print(result)

    result = px.generate_text(
        'Explain what is ProxAI and why it is useful.',
        provider_model=('openai', 'gpt-4o'))
    print(result)
```
* ⭐️ **Simple Unified API:** Pythonic, easy, intuitive, and unified API for all AI models connection.
* 💡 **All Major AI Providers:** Including Gemini, OpenAI, Claude, Grok, DeepSeek, Cohere, Mistral, and more.
* 🚀 **Always Up-to-Date:** Let ProxAI handle all new released model integrations, you just list and pick.
* 💻 **Model-agnostic AI development:** Write python code without thinking which AI provider and model you will use.


## Features

* ⏰ **Ready to Start:** Takes 2 minutes to connect and get responses from all major models.
* 🔍 **Pick and Switch:** Experiment with different models to find the best fit.
* 💾 **Caching:** Speed up responses and reduce costs with built-in query and model caching.
* 💰 **Cost Estimation:** Basic tracking of your estimated API call costs.

## Quick Start

Check out our [Quick Start Guide](https://www.proxai.co/proxai-docs/quick-start) for a step-by-step guide on how to get started with ProxAI.

1.  **Install ProxAI:**
    ```bash
    pip install proxai
    ```

2.  **Set API Keys:**
    Export your AI provider API keys as environment variables:
    ```bash
    export OPENAI_API_KEY="your-openai-key"
    export GEMINI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    # Add other provider keys as needed
    ```
    See [Provider Integrations](https://www.proxai.co/proxai-docs/provider-integrations) page.

3.  **Basic Usage:**
    ```python
    import proxai as px

    # Write model agnostic function
    def meaning_of_universe():
      response = px.generate_text('What is the meaning of universe?')
      print(response)

    # List available models
    provider_models = px.models.list_models()

    # Generate response for each model
    for provider_model in provider_models:
      px.set_model(provider_model)
      print(f"Testing {provider_model} model")
      meaning_of_universe()
    ```

## 📚 Documentation

For full details on installation, all features, and advanced usage, please visit our **[Documentation](https://www.proxai.co/proxai-docs)**.

## 📈 ProxDash (Optional Dashboard)

Enhance your ProxAI experience with [ProxDash](https://www.proxai.co/pricing), our optional monitoring platform for usage tracking, analytics, and experiment management. The ProxAI library works perfectly standalone.

## 🤝 Contribute & Connect

* **Community:** [Learn how to contribute](https://www.proxai.co/resources/community)
* **Discord:** [Join our server](https://discord.gg/Hxg6tPpX)
* **Issues:** [Report bugs or request features](https://github.com/proxai/proxai/issues)
* **GitHub:** <https://github.com/proxai/proxai>

## ⚖️ License

MIT License. See [LICENSE](https://github.com/proxai/proxai/LICENSE) for details.

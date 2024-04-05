# ProxAI

ProxAI is a lightweight abstraction layer for foundational AI model connections.
It enables easy switching between providers and models for benchmarking or any
other purposes. ProxAI adheres to the Zen of Python principles and prioritizes
simplicity.

## Installation

### Package Installation

The simplest way to get ProxAI is via pip:

```bash
$ pip install proxai
```

### Foundation Model API Keys
Before using ProxAI, set up access to third-party model providers as many as
you wish. The principle is that more is better. Add your keys to `~/.zshrc`
and run `source ~/.zshrc` before running ProxAI.

   *  **OpenAI:**
      * Get API key from: https://platform.openai.com/api-keys
      * Environment variable: `export OPENAI_API_KEY="your-key"`

   *  **Claude:**
      * Get API key from: https://console.anthropic.com/settings/keys
      * Environment variable: `export ANTHROPIC_API_KEY="your-key"`

   *  **Gemini:**
      * Get API key from: https://aistudio.google.com/app/apikey
      * Environment variable: `export GOOGLE_API_KEY="your-key"`

   *  **Cohere:**
      * Get API key from: https://dashboard.cohere.com/api-keys
      * Environment variable: `export CO_API_KEY="your-key"`

   *  **Databricks:**
      * Create workspace from: https://accounts.cloud.databricks.com/workspaces
      * In your Databricks workspace, click your *Databricks username* in the
      top bar, and then select *User Settings* from the drop down. Click
      *Developer*. Next to *Access tokens*, click *Manage*. Click *Generate*
      new token. [Official Documentation](https://docs.databricks.com/en/dev-tools/auth/pat.html)
      * Add token to environment variable as: `export DATABRICKS_TOKEN="your-key"`
      * Add your workspace url address to environment variable as:
      `export DATABRICKS_HOST='https://<your-workspace-id>.cloud.databricks.com`
      * Be careful about format. For example, ending workspace url as
      `.cloud.databricks.com/` gives an error.

   *  **Mistral:**
      * Get API key from: https://console.mistral.ai/api-keys/
      * Environment variable: `export MISTRAL_API_KEY="your-key"`

   *  **Hugging Face:**
      * Get API key from: https://huggingface.co/settings/tokens
      * Environment variable: `export HUGGINGFACE_API_KEY="your-key"`
      * Sign terms and conditions on: https://huggingface.co/google/gemma-7b-it
      * **Note:** Registered models are not working well.

## Usage

To-Do: "example code", "documents", "examples".

## Contributing to ProxAI

We are looking for contributors! We welcome all kinds of contributors, from fixing small typos to implementing bigger features. While we are working on a simple community guideline, don't wait for it. Feel free to jump in and contribute.

To be able to run on your local machine, follow these commands:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ git clone https://github.com/proxai/proxai.git
$ cd proxai
$ pip install poetry
$ poetry install
$ python3 examples/ask_about_model.py
```

Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`proxai` was created by github@proxai.co. github@proxai.co retains all rights to the source and it may not be reproduced, distributed, or used to create derivative works.

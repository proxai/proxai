import dataclasses
import enum
from typing import Union


class RunType(enum.Enum):
  PRODUCTION = 1
  TEST = 2


class Provider(str, enum.Enum):
  OPENAI = 'openai'
  CLAUDE = 'claude'
  GEMINI = 'gemini'
  COHERE = 'cohere'
  DATABRICKS = 'databricks'
  MISTRAL = 'mistral'
  HUGGING_FACE = 'hugging_face'


class OpenAIModel(str, enum.Enum):
  """OpenAI models.

  Models provided by OpenAI:
  https://platform.openai.com/docs/guides/text-generation
  """
  # Newer models (2023–)
  GPT_4 = 'gpt-4'
  GPT_4_TURBO_PREVIEW = 'gpt-4-turbo-preview'
  GPT_3_5_TURBO = 'gpt-3.5-turbo'

  # Updated legacy models (2023)
  GPT_3_5_TURBO_INSTRUCT = 'gpt-3.5-turbo-instruct'
  BABBAGE_002 = 'babbage-002'
  DAVINCI_002 = 'davinci-002'


class ClaudeModel(str, enum.Enum):
  """Claude models.

  Models provided by Claude:
  https://claude.ai/docs/models
  """
  # Latest models (03/28/2024)
  CLAUDE_3_OPUS =  'claude-3-opus-20240229'
  CLAUDE_3_SONNET = 'claude-3-sonnet-20240229'
  CLAUDE_3_HAIKU = 'claude-3-haiku-20240307'


class GeminiModel(str, enum.Enum):
  """Gemini models.

  Models provided by Gemini:
  https://ai.google.dev/models/gemini
  """
  GEMINI_1_0_PRO = 'models/gemini-1.0-pro'
  GEMINI_1_0_PRO_001 =  'models/gemini-1.0-pro-001'
  GEMINI_1_0_PRO_LATEST = 'models/gemini-1.0-pro-latest'
  GEMINI_1_0_PRO_VISION_LATEST = 'models/gemini-1.0-pro-vision-latest'
  GEMINI_PRO = 'models/gemini-pro'
  GEMINI_PRO_VISION = 'models/gemini-pro-vision'


class CohereModel(str, enum.Enum):
  """Cohere models.

  Models provided by Cohere:
  https://docs.cohere.com/docs/models
  """
  COMMAND_LIGHT = 'command-light'
  COMMAND_LIGHT_NIGHTLY = 'command-light-nightly'
  COMMAND = 'command'
  COMMAND_NIGHTLY = 'command-nightly'
  COMMAND_R = 'command-r'


class DatabricksModel(str, enum.Enum):
  """Databricks models.

  Models provided by Databricks:
  https://docs.databricks.com/en/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis
  """
  DATABRICKS_DBRX_INSTRUCT = 'databricks-dbrx-instruct'
  DATABRICKS_MIXTRAL_8x7b_INSTRUCT = 'databricks-mixtral-8x7b-instruct'
  DATABRICKS_LLAMA_2_70b_CHAT = 'databricks-llama-2-70b-chat'


class MistralModel(str, enum.Enum):
  """Mistral models.

  Models provided by Mistral:
  https://docs.mistral.ai/platform/endpoints/
  """
  OPEN_MISTRAL_7B = 'open-mistral-7b'
  OPEN_MIXTRAL_8X7B = 'open-mixtral-8x7b'
  MISTRAL_SMALL_LATEST = 'mistral-small-latest'
  MISTRAL_MEDIUM_LATEST = 'mistral-medium-latest'
  MISTRAL_LARGE_LATEST = 'mistral-large-latest'


class HuggingFaceModel(str, enum.Enum):
  """Hugging Face models.

  Models provided by Hugging Face on HuggingFaceChat:
  https://huggingface.co/chat/models
  """
  # To be able to use Google models, you need to sign terms of service:
  GOOGLE_GEMMA_7B_IT = 'google/gemma-7b-it'
  # Requires pro subscription:
  # META_LLAMA_2_70B_CHAT_HF = 'meta-llama/Llama-2-70b-chat-hf'
  # Requires pro subscription:
  # CODELLAMA_70B_INSTRUCT_HF = 'codellama/CodeLlama-70b-Instruct-hf'
  MISTRAL_MIXTRAL_8X7B_INSTRUCT_V01 = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
  MISTRAL_MISTRAL_7B_INSTRUCT_V02 = 'mistralai/Mistral-7B-Instruct-v0.2'
  NOUS_HERMES_2_MIXTRAL_8X7B_DPO = 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'
  OPENCHAT_3_5 = 'openchat/openchat-3.5-0106'


MODEL_MAP = {
    Provider.OPENAI: OpenAIModel,
    Provider.CLAUDE: ClaudeModel,
    Provider.GEMINI: GeminiModel,
    Provider.COHERE: CohereModel,
    Provider.DATABRICKS: DatabricksModel,
    Provider.MISTRAL: MistralModel,
    Provider.HUGGING_FACE: HuggingFaceModel,
}


GENERATE_TEXT_MODELS = {
    Provider.OPENAI: [
        OpenAIModel.GPT_4,
        OpenAIModel.GPT_4_TURBO_PREVIEW,
        OpenAIModel.GPT_3_5_TURBO,
    ],
    Provider.CLAUDE: [
        ClaudeModel.CLAUDE_3_OPUS,
        ClaudeModel.CLAUDE_3_SONNET,
        ClaudeModel.CLAUDE_3_HAIKU,
    ],
    Provider.GEMINI: [
        GeminiModel.GEMINI_1_0_PRO,
        GeminiModel.GEMINI_1_0_PRO_001,
        GeminiModel.GEMINI_1_0_PRO_LATEST,
        GeminiModel.GEMINI_PRO,
    ],
    Provider.COHERE: [
        CohereModel.COMMAND_LIGHT,
        CohereModel.COMMAND_LIGHT_NIGHTLY,
        CohereModel.COMMAND,
        CohereModel.COMMAND_NIGHTLY,
        CohereModel.COMMAND_R,
    ],
    Provider.DATABRICKS: [
        DatabricksModel.DATABRICKS_DBRX_INSTRUCT,
        DatabricksModel.DATABRICKS_MIXTRAL_8x7b_INSTRUCT,
        DatabricksModel.DATABRICKS_LLAMA_2_70b_CHAT,
    ],
    Provider.MISTRAL: [
        MistralModel.OPEN_MISTRAL_7B,
        MistralModel.OPEN_MIXTRAL_8X7B,
        MistralModel.MISTRAL_SMALL_LATEST,
        MistralModel.MISTRAL_MEDIUM_LATEST,
        MistralModel.MISTRAL_LARGE_LATEST,
    ],
    Provider.HUGGING_FACE: [
        HuggingFaceModel.GOOGLE_GEMMA_7B_IT,
        HuggingFaceModel.MISTRAL_MIXTRAL_8X7B_INSTRUCT_V01,
        HuggingFaceModel.MISTRAL_MISTRAL_7B_INSTRUCT_V02,
        HuggingFaceModel.NOUS_HERMES_2_MIXTRAL_8X7B_DPO,
        HuggingFaceModel.OPENCHAT_3_5,
    ],
}


@dataclasses.dataclass
class ModelSignature:
  provider: Provider
  model: Union[
      OpenAIModel,
      ClaudeModel,
      GeminiModel,
      CohereModel]

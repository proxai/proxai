from typing import Union, Optional, Type
from databricks_genai_inference import ChatCompletion
import proxai.types as types
from .databricks_mock import DatabricksMock
from .model_connector import ModelConnector


class DatabricksConnector(ModelConnector):
  def init_model(self):
    return ChatCompletion

  def init_mock_model(self):
    return DatabricksMock()

  def generate_text_proc(self, prompt: str, max_tokens: int) -> str:
    response = self.api.create(
        model=self.model_signature.model.value,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}])
    return response.message

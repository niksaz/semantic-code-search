from typing import Any, Dict, Optional

from encoders import CodeTokensASTEncoder, NBoWEncoder
from .nbow_model import Model


class NeuralASTModel(Model):
  CODE_ENCODER_TYPE = CodeTokensASTEncoder
  QUERY_ENCODER_TYPE = NBoWEncoder
  MODEL_NAME = "neuralastmodel"

  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    hypers = {}
    for key, value in cls.CODE_ENCODER_TYPE.get_default_hyperparameters().items():
      hypers[f'code_{key}'] = value
    for key, value in cls.QUERY_ENCODER_TYPE.get_default_hyperparameters().items():
      hypers[f'query_{key}'] = value
    model_hypers = {
      'code_use_subtokens': False,
      'code_mark_subtoken_end': False,
      'loss': 'cosine',
      'batch_size': 100
    }
    hypers.update(super().get_default_hyperparameters())
    hypers.update(model_hypers)
    return hypers

  def _get_model_name(self) -> str:
    return self.MODEL_NAME

  def __init__(self,
               hyperparameters: Dict[str, Any],
               run_name: str = None,
               model_save_dir: Optional[str] = None,
               log_save_dir: Optional[str] = None):
    super().__init__(
      hyperparameters,
      code_encoder_type=self.CODE_ENCODER_TYPE,
      query_encoder_type=self.QUERY_ENCODER_TYPE,
      run_name=run_name,
      model_save_dir=model_save_dir,
      log_save_dir=log_save_dir)

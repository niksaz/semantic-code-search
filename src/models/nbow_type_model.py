from typing import Any, Dict, Optional

from encoders import CodeTokensTypesEncoder
from .nbow_model import NeuralBoWModel


class NeuralBoWTypeModel(NeuralBoWModel):
  def __init__(self,
               hyperparameters: Dict[str, Any],
               run_name: str = None,
               model_save_dir: Optional[str] = None,
               log_save_dir: Optional[str] = None):
    super().__init__(
      hyperparameters=hyperparameters,
      run_name=run_name,
      model_save_dir=model_save_dir,
      log_save_dir=log_save_dir,
      code_encoder_type=CodeTokensTypesEncoder)

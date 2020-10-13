from typing import Any, Dict, Optional, Tuple, List

import tensorflow as tf

from . import Encoder, QueryType, NBoWEncoder


class CodeTokensTypesEncoder(Encoder):

  ENCODER_CLASS = NBoWEncoder

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    self.code_encoder = self.ENCODER_CLASS(label, hyperparameters, metadata)

  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    return cls.ENCODER_CLASS.get_default_hyperparameters()

  @property
  def output_representation_size(self) -> int:
    return self.code_encoder.output_representation_size

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    return self.code_encoder.make_model(is_train)

  @classmethod
  def init_metadata(cls) -> Dict[str, Any]:
    return cls.ENCODER_CLASS.init_metadata()

  @classmethod
  def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any],
                                use_subtokens: bool = False, mark_subtoken_end: bool = False) -> None:
    cls.ENCODER_CLASS.load_metadata_from_sample(
      data_to_load['code_tokens'],
      raw_metadata,
      use_subtokens,
      mark_subtoken_end)

  @classmethod
  def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                        raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return cls.ENCODER_CLASS.finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)

  @classmethod
  def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                            data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                            is_test: bool = True) -> bool:
    return cls.ENCODER_CLASS.load_data_from_sample(
      encoder_label,
      hyperparameters,
      metadata,
      data_to_load['code_tokens'],
      function_name,
      result_holder,
      is_test)

  def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
    self.code_encoder.init_minibatch(batch_data)

  def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                 query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
    return self.code_encoder.extend_minibatch_by_sample(batch_data, sample, is_train, query_type)

  def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
    self.code_encoder.minibatch_to_feed_dict(batch_data, feed_dict, is_train)

  def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
    return self.code_encoder.get_token_embeddings()

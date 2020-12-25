from typing import Any, Dict, Optional, Tuple, List

import tensorflow as tf

from . import Encoder, QueryType
from .graph_encoder import GraphEncoder
from utils import data_pipeline


class GraphTokensEncoder(Encoder):
  AST_ENCODER_CLASS = GraphEncoder
  AST_ENCODER_LABEL = 'graph_encoder'

  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    ast_hypers = cls.AST_ENCODER_CLASS.get_default_hyperparameters()
    hypers = {}
    hypers.update(ast_hypers)
    return hypers

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    self.ast_encoder = self.AST_ENCODER_CLASS(
      label,
      hyperparameters,
      metadata[self.AST_ENCODER_LABEL])

  @property
  def output_representation_size(self) -> int:
    return self.ast_encoder.output_representation_size

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    with tf.variable_scope('ast_encoder'):
      ast_encoder_tensor = self.ast_encoder.make_model(is_train)
    return ast_encoder_tensor

  @classmethod
  def init_metadata(cls) -> Dict[str, Any]:
    return {
      cls.AST_ENCODER_LABEL: cls.AST_ENCODER_CLASS.init_metadata(),
    }

  @classmethod
  def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any],
                                use_subtokens: bool = False, mark_subtoken_end: bool = False) -> None:
    cls.AST_ENCODER_CLASS.load_metadata_from_sample(
      data_to_load[data_pipeline.GRAPH_LABEL],
      raw_metadata[cls.AST_ENCODER_LABEL],
      use_subtokens,
      mark_subtoken_end)

  @classmethod
  def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                        raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    ast_encoder_metadata_list = list(map(lambda raw_metadata: raw_metadata[cls.AST_ENCODER_LABEL], raw_metadata_list))
    return {
      cls.AST_ENCODER_LABEL:
        cls.AST_ENCODER_CLASS.finalise_metadata(
          encoder_label,
          hyperparameters,
          ast_encoder_metadata_list),
    }

  @classmethod
  def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                            data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                            is_test: bool = True) -> bool:
    if cls.AST_ENCODER_LABEL not in result_holder:
      result_holder[cls.AST_ENCODER_LABEL] = {}
    use_ast_sampler = cls.AST_ENCODER_CLASS.load_data_from_sample(
      encoder_label,
      hyperparameters,
      metadata[cls.AST_ENCODER_LABEL],
      data_to_load[data_pipeline.GRAPH_LABEL],
      function_name,
      result_holder[cls.AST_ENCODER_LABEL],
      is_test)
    return use_ast_sampler

  def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
    batch_data[self.AST_ENCODER_LABEL] = {}
    self.ast_encoder.init_minibatch(batch_data[self.AST_ENCODER_LABEL])

  def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                 query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
    is_ast_batch_full = self.ast_encoder.extend_minibatch_by_sample(
      batch_data[self.AST_ENCODER_LABEL], sample[self.AST_ENCODER_LABEL], is_train, query_type)
    return is_ast_batch_full

  def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
    self.ast_encoder.minibatch_to_feed_dict(batch_data[self.AST_ENCODER_LABEL], feed_dict, is_train)

  def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
    raise NotImplementedError

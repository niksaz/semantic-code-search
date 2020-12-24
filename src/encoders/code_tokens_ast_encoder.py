import collections
from typing import Any, Dict, Optional, Tuple, List

import tensorflow as tf

from . import Encoder, QueryType, NBoWEncoder, ASTPretrainedNBoWEncoder, GraphPretrainedNBoWEncoder, TBCNNEncoder
from utils import data_pipeline
from encoders.utils import tree_processing


def get_graph_nodes(graph: collections.OrderedDict) -> List[str]:
  if graph:
    return graph['nodes']
  else:
    return ['*#$%UNKNOWN*#$%']


class DataPreprocessor:
  @staticmethod
  def extract_code_data(data_to_load):
    return data_to_load[data_pipeline.CODE_TOKENS_LABEL]

  @staticmethod
  def extract_ast_data(data_to_load):
    # node2vec [Graphs]
    return get_graph_nodes(data_to_load[data_pipeline.GRAPH_LABEL])
    # NBOW+Types [AST]
    # node2vec [AST]
    # return tree_processing.get_type_bag_from_tree(data_to_load[data_pipeline.TREE_LABEL])
    # TBCNN [AST]
    # return data_to_load[data_pipeline.TREE_LABEL]


class CodeTokensASTEncoder(Encoder):
  CODE_ENCODER_CLASS = NBoWEncoder
  # node2vec [Graphs]
  AST_ENCODER_CLASS = GraphPretrainedNBoWEncoder
  # NBOW+Types [AST]
  # AST_ENCODER_CLASS = NBoWEncoder
  # node2vec [AST]
  # AST_ENCODER_CLASS = ASTPretrainedNBoWEncoder
  # TBCNN [AST]
  # AST_ENCODER_CLASS = TBCNNEncoder
  CODE_ENCODER_LABEL = 'code_encoder'
  AST_ENCODER_LABEL = 'ast_encoder'

  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    code_hypers = cls.CODE_ENCODER_CLASS.get_default_hyperparameters()
    ast_hypers = cls.AST_ENCODER_CLASS.get_default_hyperparameters()
    for key, value in ast_hypers.items():
      if key in code_hypers and code_hypers[key] != value:
        raise AssertionError(
          f'The same hyperparameter is set differently for code {code_hypers[key]} and for ast {value}.')
    hypers = {}
    hypers.update(code_hypers)
    hypers.update(ast_hypers)
    return hypers

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    self.code_encoder = self.CODE_ENCODER_CLASS(
      label,
      hyperparameters,
      metadata[self.CODE_ENCODER_LABEL])
    self.ast_encoder = self.AST_ENCODER_CLASS(
      label,
      hyperparameters,
      metadata[self.AST_ENCODER_LABEL])

  @property
  def output_representation_size(self) -> int:
    assert self.code_encoder.output_representation_size == self.ast_encoder.output_representation_size
    return self.code_encoder.output_representation_size

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    with tf.variable_scope('code_tokens_encoder'):
      code_encoder_tensor = self.code_encoder.make_model(is_train)
    with tf.variable_scope('ast_encoder'):
      ast_encoder_tensor = self.ast_encoder.make_model(is_train)
    return code_encoder_tensor + ast_encoder_tensor

  @classmethod
  def init_metadata(cls) -> Dict[str, Any]:
    return {
      cls.CODE_ENCODER_LABEL: cls.CODE_ENCODER_CLASS.init_metadata(),
      cls.AST_ENCODER_LABEL: cls.AST_ENCODER_CLASS.init_metadata(),
    }

  @classmethod
  def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any],
                                use_subtokens: bool = False, mark_subtoken_end: bool = False) -> None:
    cls.CODE_ENCODER_CLASS.load_metadata_from_sample(
      DataPreprocessor.extract_code_data(data_to_load),
      raw_metadata[cls.CODE_ENCODER_LABEL],
      use_subtokens,
      mark_subtoken_end)
    cls.AST_ENCODER_CLASS.load_metadata_from_sample(
      DataPreprocessor.extract_ast_data(data_to_load),
      raw_metadata[cls.AST_ENCODER_LABEL],
      use_subtokens,
      mark_subtoken_end)

  @classmethod
  def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                        raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    code_encoder_metadata_list = list(map(lambda raw_metadata: raw_metadata[cls.CODE_ENCODER_LABEL], raw_metadata_list))
    ast_encoder_metadata_list = list(map(lambda raw_metadata: raw_metadata[cls.AST_ENCODER_LABEL], raw_metadata_list))
    return {
      cls.CODE_ENCODER_LABEL:
        cls.CODE_ENCODER_CLASS.finalise_metadata(
          encoder_label,
          hyperparameters,
          code_encoder_metadata_list),
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
    if cls.CODE_ENCODER_LABEL not in result_holder:
      result_holder[cls.CODE_ENCODER_LABEL] = {}
    use_code_sample = cls.CODE_ENCODER_CLASS.load_data_from_sample(
      encoder_label,
      hyperparameters,
      metadata[cls.CODE_ENCODER_LABEL],
      DataPreprocessor.extract_code_data(data_to_load),
      function_name,
      result_holder[cls.CODE_ENCODER_LABEL],
      is_test)
    if cls.AST_ENCODER_LABEL not in result_holder:
      result_holder[cls.AST_ENCODER_LABEL] = {}
    use_ast_sampler = cls.AST_ENCODER_CLASS.load_data_from_sample(
      encoder_label,
      hyperparameters,
      metadata[cls.AST_ENCODER_LABEL],
      DataPreprocessor.extract_ast_data(data_to_load),
      function_name,
      result_holder[cls.AST_ENCODER_LABEL],
      is_test)
    return use_code_sample and use_ast_sampler

  def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
    batch_data[self.CODE_ENCODER_LABEL] = {}
    self.code_encoder.init_minibatch(batch_data[self.CODE_ENCODER_LABEL])
    batch_data[self.AST_ENCODER_LABEL] = {}
    self.ast_encoder.init_minibatch(batch_data[self.AST_ENCODER_LABEL])

  def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                 query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
    is_code_batch_full = self.code_encoder.extend_minibatch_by_sample(
      batch_data[self.CODE_ENCODER_LABEL], sample[self.CODE_ENCODER_LABEL], is_train, query_type)
    is_ast_batch_full = self.ast_encoder.extend_minibatch_by_sample(
      batch_data[self.AST_ENCODER_LABEL], sample[self.AST_ENCODER_LABEL], is_train, query_type)
    assert is_code_batch_full == is_ast_batch_full
    return is_code_batch_full

  def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
    self.code_encoder.minibatch_to_feed_dict(batch_data[self.CODE_ENCODER_LABEL], feed_dict, is_train)
    self.ast_encoder.minibatch_to_feed_dict(batch_data[self.AST_ENCODER_LABEL], feed_dict, is_train)

  def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
    raise NotImplementedError

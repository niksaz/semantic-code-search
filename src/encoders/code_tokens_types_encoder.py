import re
from typing import Any, Dict, Optional, Tuple, List

import tensorflow as tf

from . import Encoder, QueryType, NBoWEncoder
from utils import data_pipeline


def _linearize_tree(node: data_pipeline.TreeNode, linearization: List[data_pipeline.TreeNode]):
  linearization.append(node)
  for child in node['children']:
    _linearize_tree(child, linearization)


def _get_code_tokens_from_tree(tree: data_pipeline.TreeNode) -> List[str]:
  linearization = []
  _linearize_tree(tree, linearization)
  node_tokens = list(map(lambda node: node['string'], linearization))
  python_identifier_pattern = re.compile(r'^[^\d\W]\w*\Z', re.UNICODE)
  code_tokens = list(filter(lambda token: re.match(python_identifier_pattern, token), node_tokens))
  return code_tokens


def _get_type_bag_from_tree(tree: data_pipeline.TreeNode) -> List[str]:
  linearization = []
  _linearize_tree(tree, linearization)
  type_tokens = list(map(lambda node: node['type'], linearization))
  type_bag = set(type_tokens)
  return list(type_bag)


class CodeTokensTypesEncoder(Encoder):
  CODE_ENCODER_CLASS = NBoWEncoder
  TYPE_ENCODER_CLASS = NBoWEncoder
  CODE_ENCODER_LABEL = 'code_encoder'
  TYPE_ENCODER_LABEL = 'type_encoder'

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    self.code_encoder = self.CODE_ENCODER_CLASS(
      label,
      hyperparameters,
      metadata[self.CODE_ENCODER_LABEL])
    self.type_encoder = self.TYPE_ENCODER_CLASS(
      label,
      hyperparameters,
      metadata[self.TYPE_ENCODER_LABEL])

  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    return {}

  @property
  def output_representation_size(self) -> int:
    assert self.code_encoder.output_representation_size == self.type_encoder.output_representation_size
    return self.code_encoder.output_representation_size

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    with tf.variable_scope("code_encoder"):
      code_encoder_tensor = self.code_encoder.make_model(is_train)
    with tf.variable_scope("type_encoder"):
      type_encoder_tensor = self.type_encoder.make_model(is_train)
    return code_encoder_tensor + type_encoder_tensor

  @classmethod
  def init_metadata(cls) -> Dict[str, Any]:
    return {
      cls.CODE_ENCODER_LABEL: cls.CODE_ENCODER_CLASS.init_metadata(),
      cls.TYPE_ENCODER_LABEL: cls.TYPE_ENCODER_CLASS.init_metadata(),
    }

  @classmethod
  def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any],
                                use_subtokens: bool = False, mark_subtoken_end: bool = False) -> None:
    cls.CODE_ENCODER_CLASS.load_metadata_from_sample(
      data_to_load[data_pipeline.CODE_TOKENS_LABEL],
      raw_metadata[cls.CODE_ENCODER_LABEL],
      use_subtokens,
      mark_subtoken_end)
    cls.TYPE_ENCODER_CLASS.load_metadata_from_sample(
      _get_type_bag_from_tree(data_to_load[data_pipeline.RAW_TREE_LABEL]),
      raw_metadata[cls.TYPE_ENCODER_LABEL],
      use_subtokens,
      mark_subtoken_end)

  @classmethod
  def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                        raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    code_encoder_metadata_list = list(map(lambda raw_metadata: raw_metadata[cls.CODE_ENCODER_LABEL], raw_metadata_list))
    type_encoder_metadata_list = list(map(lambda raw_metadata: raw_metadata[cls.TYPE_ENCODER_LABEL], raw_metadata_list))
    return {
      cls.CODE_ENCODER_LABEL:
        cls.CODE_ENCODER_CLASS.finalise_metadata(
          encoder_label,
          hyperparameters,
          code_encoder_metadata_list),
      cls.TYPE_ENCODER_LABEL:
        cls.TYPE_ENCODER_CLASS.finalise_metadata(
          encoder_label,
          hyperparameters,
          type_encoder_metadata_list),
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
      data_to_load[data_pipeline.CODE_TOKENS_LABEL],
      function_name,
      result_holder[cls.CODE_ENCODER_LABEL],
      is_test)
    if cls.TYPE_ENCODER_LABEL not in result_holder:
      result_holder[cls.TYPE_ENCODER_LABEL] = {}
    use_type_sample = cls.TYPE_ENCODER_CLASS.load_data_from_sample(
      encoder_label,
      hyperparameters,
      metadata[cls.TYPE_ENCODER_LABEL],
      _get_type_bag_from_tree(data_to_load[data_pipeline.RAW_TREE_LABEL]),
      function_name,
      result_holder[cls.TYPE_ENCODER_LABEL],
      is_test)
    return use_code_sample and use_type_sample

  def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
    batch_data[self.CODE_ENCODER_LABEL] = {}
    self.code_encoder.init_minibatch(batch_data[self.CODE_ENCODER_LABEL])
    batch_data[self.TYPE_ENCODER_LABEL] = {}
    self.type_encoder.init_minibatch(batch_data[self.TYPE_ENCODER_LABEL])

  def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                 query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
    is_code_batch_full = self.code_encoder.extend_minibatch_by_sample(
      batch_data[self.CODE_ENCODER_LABEL], sample[self.CODE_ENCODER_LABEL], is_train, query_type)
    is_type_batch_full = self.type_encoder.extend_minibatch_by_sample(
      batch_data[self.TYPE_ENCODER_LABEL], sample[self.TYPE_ENCODER_LABEL], is_train, query_type)
    assert is_code_batch_full == is_type_batch_full
    return is_code_batch_full

  def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
    self.code_encoder.minibatch_to_feed_dict(batch_data[self.CODE_ENCODER_LABEL], feed_dict, is_train)
    self.type_encoder.minibatch_to_feed_dict(batch_data[self.TYPE_ENCODER_LABEL], feed_dict, is_train)

  def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
    raise NotImplementedError

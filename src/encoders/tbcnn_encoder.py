import collections
from typing import Dict, Any, Tuple, List, Optional

import tensorflow as tf
import numpy as np

from .ast_encoder import ASTEncoder, _try_to_queue_node
from .utils import tbcnn_network
from .utils import tree_processing
from utils import data_pipeline
from utils import tfutils
from utils.tfutils import pool_sequence_embedding


class TBCNNEncoder(ASTEncoder):
  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    hypers = super().get_default_hyperparameters()
    return hypers

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    assert label == 'code', 'TBCNNEncoder should only be used for code'

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    with tf.variable_scope('tbcnn_encoder'):
      self._make_placeholders()

      node_tokens = self.token_embedding_layer(self.placeholders['node_token_ids'])
      node_token_masks = self.placeholders['node_masks']
      node_token_lens = tf.reduce_sum(node_token_masks, axis=1)  # B
      token_encoding = pool_sequence_embedding('mean',
                                               sequence_token_embeddings=node_tokens,
                                               sequence_lengths=node_token_lens,
                                               sequence_token_masks=node_token_masks)
      node_types = self.type_embedding_layer(self.placeholders['node_type_ids'])
      children = self.placeholders['children']
      type_encoding = tbcnn_network.init_net(node_types, children, self.get_hyper('type_embedding_size'))
    return token_encoding + type_encoding

  def _make_placeholders(self):
    super()._make_placeholders()
    self.placeholders['node_masks'] = tf.placeholder(tf.float32, shape=[None, None], name='node_masks')
    self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, shape=[None, None], name='node_token_ids')
    self.placeholders['node_type_ids'] = tf.placeholder(tf.int32, shape=[None, None], name='node_type_ids')
    self.placeholders['children'] = tf.placeholder(tf.int32, shape=(None, None, None), name='children')

  @classmethod
  def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                            data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                            is_test: bool = True) -> bool:
    nodes, children = tree_processing.linearize_tree_bfs(
      data_to_load,
      hyperparameters[f'{encoder_label}_max_num_nodes']
    )
    node_tokens = [node['string'] for node in nodes]
    node_token_ids, mask = (
      tfutils.convert_and_pad_token_sequence(
        metadata['token_vocab'],
        node_tokens,
        hyperparameters[f'{encoder_label}_max_num_tokens']))
    result_holder[f'{encoder_label}_node_masks'] = list(mask)
    result_holder[f'{encoder_label}_node_token_ids'] = list(node_token_ids)
    node_types = [node['type'] for node in nodes]
    n = len(node_types)
    node_type_ids, mask = (
      tfutils.convert_and_pad_token_sequence(
        metadata['type_vocab'],
        node_types,
        n))
    assert len(node_type_ids) == n
    assert np.all(mask == 1)
    result_holder[f'{encoder_label}_node_type_ids'] = list(node_type_ids)
    result_holder[f'{encoder_label}_children'] = children
    return True

  def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
    super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
    node_masks = batch_data['node_masks']
    node_token_ids = batch_data['node_token_ids']
    node_type_ids = batch_data['node_type_ids']
    children = batch_data['children']
    node_masks, node_token_ids, node_type_ids, children = tbcnn_network.pad_batch(
      node_masks, node_token_ids, node_type_ids, children
    )
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_masks'], node_masks)
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_token_ids'], node_token_ids)
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_type_ids'], node_type_ids)
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['children'], children)
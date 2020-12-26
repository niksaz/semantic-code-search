import collections
from typing import Dict, Any, Tuple, List, Optional

import tensorflow as tf
import numpy as np

from .ast_encoder import ASTEncoder, _try_to_queue_node
from .utils import tbcnn_network
from utils import data_pipeline
from utils import tfutils


def _linearize_tree_bfs(
    root: data_pipeline.TreeNode,
    max_nodes: int = -1) -> Tuple[List[data_pipeline.TreeNode], List[List[int]]]:
  nodes: List[data_pipeline.TreeNode] = []
  children: List[List[int]] = []
  node_queue = collections.deque()
  nodes_queued = 0
  nodes_queued += _try_to_queue_node(root, node_queue, nodes_queued, max_nodes)
  while node_queue:
    node = node_queue.popleft()
    node_children: List[int] = []
    for child in node['children']:
      if _try_to_queue_node(child, node_queue, nodes_queued, max_nodes):
        node_children.append(nodes_queued)
        nodes_queued += 1
    nodes.append(node)
    children.append(node_children)
  return nodes, children


class TBCNNEncoder(Encoder):
  @classmethod
  def get_default_hyperparameters(cls) -> Dict[str, Any]:
    encoder_hypers = {
      'max_num_nodes': 100
    }
    hypers = super().get_default_hyperparameters()
    hypers.update(encoder_hypers)
    return hypers

  def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
    super().__init__(label, hyperparameters, metadata)
    assert label == 'code', 'TBCNNEncoder should only be used for code'

  def _make_placeholders(self):
    super()._make_placeholders()
    self.placeholders['node_type_ids'] = tf.placeholder(tf.int32, shape=[None, None], name='node_type_ids')
    self.placeholders['children'] = tf.placeholder(tf.int32, shape=(None, None, None), name='children')

  def make_model(self, is_train: bool = False) -> tf.Tensor:
    with tf.variable_scope('tbcnn_encoder'):
      self._make_placeholders()

      nodes = self.embedding_layer(self.placeholders['node_type_ids'])
      children = self.placeholders['children']
      hidden = tbcnn_network.init_net(nodes, children, self.get_hyper('type_embedding_size'))
    return hidden

  @classmethod
  def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                            data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                            is_test: bool = True) -> bool:
    nodes, children = _linearize_tree_bfs(data_to_load, hyperparameters[f'{encoder_label}_max_num_nodes'])
    n = len(nodes)
    node_types = [node['type'] for node in nodes]
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
    node_type_ids = batch_data['node_type_ids']
    children = batch_data['children']
    node_type_ids, children = tbcnn_network.pad_batch(node_type_ids, children)
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_type_ids'], node_type_ids)
    tfutils.write_to_feed_dict(feed_dict, self.placeholders['children'], children)

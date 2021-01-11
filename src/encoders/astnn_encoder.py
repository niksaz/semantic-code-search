import collections
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import tensorflow as tf

import encoders.utils.tree_processing
from utils import data_pipeline
from utils import tfutils
from .ast_encoder import ASTEncoder, _try_to_queue_node
from .utils import astnn_network

STATEMENTS = ['function_definition', 'if_statement', 'while_statement', 'do_statement', 'switch_statement',
              'compound_statement', 'for_statement']

BRACES = ['(', ')', '{', '}', '[', ']']


def _lineraize_non_block_tree_bfs(root: encoders.utils.tree_processing.TreeNode, nodes_queued: int,
                                  max_nodes: int = -1) -> Tuple[List[encoders.utils.tree_processing.TreeNode], List[List[int]]]:
    nodes: List[encoders.utils.tree_processing.TreeNode] = []
    children: List[List[int]] = []
    node_queue = collections.deque()
    nodes_queued += _try_to_queue_node(root, node_queue, nodes_queued, max_nodes)
    while node_queue:
        node = node_queue.popleft()
        if node is str:
            nodes.append(node)
            children.append([])
        node_children: List[int] = []
        for child in node['children']:
            if child in BRACES or child in STATEMENTS:
                continue
            if _try_to_queue_node(child, node_queue, nodes_queued, max_nodes):
                node_children.append(nodes_queued)
                nodes_queued += 1
        nodes.append(node)
        children.append(node_children)
    return nodes, children


def _get_tree_blocks(root: encoders.utils.tree_processing.TreeNode) -> List[encoders.utils.tree_processing.TreeNode]:
    blocks: List[encoders.utils.tree_processing.TreeNode] = []
    node_queue = collections.deque()
    node_queue.append(root)
    while node_queue:
        node = node_queue.popleft()
        if node is str:
            blocks.append(node)
            continue
        for child in node['children']:
            if child in STATEMENTS:
                node_queue.append(child)
        if node['type'] == 'compound_statement':
            node_queue.append('end')
        blocks.append(node)
    return blocks


def _linearize_and_split_tree_bfs(
        root: encoders.utils.tree_processing.TreeNode,
        max_nodes: int = -1) -> Tuple[List[List[encoders.utils.tree_processing.TreeNode]], List[List[List[int]]]]:
    nodes: List[List[encoders.utils.tree_processing.TreeNode]] = []
    children: List[List[List[int]]] = []
    blocks = _get_tree_blocks(root)
    nodes_queued = 0
    for b in blocks:
        bnodes, bchildren = _lineraize_non_block_tree_bfs(b, nodes_queued, max_nodes)
        nodes.append(bnodes)
        children.append(bchildren)
        if max_nodes != -1 and nodes_queued >= max_nodes:
            break
    return nodes, children


class ASTNNEncoder(ASTEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'tree_encoder_size': 128,
            # 'tree_hidden_size': 100
            # we're considering that hidden_size = type_embedding_size // 2,
            # so that final representation size (after bigru) would be equal to embedding_size
            'max_num_nodes': 100,
            'max_children': 100
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)
        assert label == 'code', 'ASTNNEncoder should only be used for code'

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope('astnn_encoder'):
            self._make_placeholders()

            node_types = self.embedding_layer(self.placeholders['node_type_ids'])
            children = self.placeholders['children']
            type_encoding = astnn_network.init_net(node_types, children, self.get_hyper('type_embedding_size'),
                                                   self.get_hyper('tree_encoder_size'))
        return type_encoding

    def _make_placeholders(self):
        super()._make_placeholders()
        self.placeholders['node_type_ids'] = tf.placeholder(tf.int32,
                                                            shape=[None, None, self.get_hyper('max_num_nodes')],
                                                            name='node_type_ids')
        self.placeholders['children'] = tf.placeholder(tf.int32,
                                                       shape=(None, None, self.get_hyper('max_num_nodes'), None),
                                                       name='children')

    @classmethod
    def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                              data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        node_types, children = _linearize_and_split_tree_bfs(
            data_to_load, hyperparameters[f'{encoder_label}_max_num_nodes']
        )

        def convert_and_pad(nodes_):
            n = len(nodes_)
            node_types = [node['type'] for node in nodes_]
            node_type_ids, mask = (
                tfutils.convert_and_pad_token_sequence(
                    metadata['type_vocab'],
                    node_types,
                    n))
            assert len(node_type_ids) == n
            assert np.all(mask == 1)
            return list(node_type_ids)

        node_type_ids = [convert_and_pad(nodes_) for nodes_ in node_types]
        result_holder[f'{encoder_label}_node_type_ids'] = list(node_type_ids)
        result_holder[f'{encoder_label}_children'] = children
        return True

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any],
                               is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        node_type_ids = batch_data['node_type_ids']
        children = batch_data['children']
        node_type_ids, children = astnn_network.pad_batch(
            node_type_ids, children, self.get_hyper('max_num_nodes')
        )
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_type_ids'], node_type_ids)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['children'], children)

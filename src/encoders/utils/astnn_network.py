import math
import tensorflow as tf


def init_net(nodes, children, feature_size):
  """Initialize an empty network."""

  with tf.name_scope('network'):
    tree_enc = tree_encoder_layer(feature_size, nodes, children)
    hidden = None

  with tf.name_scope('summaries'):
    tf.summary.scalar('split_size', tf.shape(nodes)[1])
    tf.summary.scalar('tree_size', tf.shape(nodes)[2])
    tf.summary.scalar('child_size', tf.shape(children)[3])
    tf.summary.histogram('logits', hidden)
    tf.summary.image('inputs', tf.expand_dims(nodes, axis=3))

  return hidden


def tree_encoder_layer(features_size, nodes, children):
  pass


def pad_batch(node_type_ids, children):
  if not node_type_ids:
    return [], []

  max_splits = max([len(b) for b in node_type_ids])
  node_type_ids = [b + [[]] * (max_splits - len(b)) for b in node_type_ids]

  max_nodes = max([len(s) for b in node_type_ids for s in b])
  node_type_ids = [[s + [-1] * (max_nodes - len(s)) for s in b] for b in node_type_ids]

  max_splits = max([len(b) for b in children])
  children = [b + [[[]]] * (max_splits - len(b)) for b in children]

  max_nodes = max([len(s) for b in children for s in b])
  children = [[s + [[]] * (max_nodes - len(s)) for s in b] for b in children]

  max_children = max([len(n) for b in children for s in b for n in s])
  children = [[[n + [0] * (max_children - len(n)) for n in s] for s in b] for b in children]

  return node_type_ids, children
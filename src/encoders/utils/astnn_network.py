import math
import tensorflow as tf


def init_net(nodes, children, feature_size, encoder_size):
    """Initialize an empty network."""

    with tf.name_scope('network'):
        # (batch_size x split_size x encoder_size)
        enc = encoder_layer(nodes, children, feature_size, encoder_size)

        # (batch_size x split_size x feature_size)
        gru_out = bigru_layer(enc, encoder_size, feature_size // 2, 1)
        # (batch_size x feature_size x split_size)
        gru_out = tf.transpose(gru_out, perm=[0, 2, 1])

        with tf.name_scope('pooling'):
            # (batch_size x feature_size)
            hidden = tf.reduce_max(gru_out, axis=1)

    with tf.name_scope('summaries'):
        tf.summary.scalar('split_size', tf.shape(nodes)[1])
        tf.summary.scalar('tree_size', tf.shape(nodes)[2])
        tf.summary.scalar('child_size', tf.shape(children)[3])
        tf.summary.histogram('logits', hidden)
        tf.summary.image('inputs', tf.expand_dims(nodes, axis=4))
        tf.summary.image('enc', tf.expand_dims(enc, axis=4))
        tf.summary.image('bigru', tf.expand_dims(gru_out, axis=4))

    return hidden


def bigru_layer(encodes, input_dim, hidden_dim, num_layers):
    batch_size = tf.shape(encodes)[0]
    hidden = tf.Variable(tf.zeros((num_layers * 2, batch_size, hidden_dim)))
    # bigru i can't even omg


def encoder_layer(nodes, children, features_size, encoder_size):
    with tf.name_scope('encoder_layer'):
        with tf.name_scope('encoder_layer'):
            batch_size, split_size = tf.shape(nodes)[0], tf.shape(nodes)[1]
            splits, split_children = [], []
            for batch_id in range(batch_size):
                for split_id in range(split_size):
                    splits.append(nodes[batch_id][split_id])
                    split_children.append(children[batch_id][split_id])

        with tf.name_scope('encoder_mul'):
            node_list = []
            traverse_mul(splits, split_children, node_list,
                         features_size, encoder_size)

        with tf.name_scope('encoder_pooling'):
            node_list = tf.stack(node_list)
            # (batch_size * split_size x encoder_size)
            enc_node = tf.reduce_max(node_list, axis=0)

        with tf.name_scope('encoder_outputs'):
            seq, start, end = [], 0, 0
            for i in range(batch_size):
                end += len(nodes[i])
                seq.append(enc_node[start:end])
                start = end
            enc_node = tf.concat(seq, axis=0, name='splits_encodings')
            # (batch_size x split_size x encoder_size)
            enc_node = tf.reshape(enc_node, (batch_size, split_size, encoder_size))
        return enc_node


def traverse_mul(nodes, children, node_list, features_size, encoder_size):
    with tf.name_scope('split_encoder_node'):
        batch_size = tf.shape(nodes)[0]
        nodes_size = tf.shape(nodes)[1]
        for j in range(nodes_size - 1, -1, -1):
            # (batch_size x 1 x max_children x encoder_size)
            current_children = tf.stack(
                [[tf.stack(
                    [W_c(nodes[i, c, :], features_size, encoder_size) for c in children[i][j]]
                )] for i in range(batch_size)],
                name='children_vectors'
            )

            # (batch_size x 1 x encoder_size)
            children_sum = tf.reduce_sum(current_children, axis=2, name='children_sum')
            # (batch_size x encoder_size)
            new_node_vector = tf.reduce_sum(
                tf.concat([nodes[:, j:j+1, :], children_sum], axis=1),
                axis=1, name='new_node_enc'
            )
            nodes[:, j, :] = new_node_vector
            node_list.append(new_node_vector)


def W_c(encoded, input_size, output_size):
    with tf.name_scope('W_c'):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / input_size))
        # init = tf.zeros([output_size,])
        biases = tf.Variable(init, name='biases')

        with tf.name_scope('summaries'):
            tf.summary.histogram('weights', [weights])
            tf.summary.histogram('biases', [biases])

        return tf.matmul(encoded, weights) + biases


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
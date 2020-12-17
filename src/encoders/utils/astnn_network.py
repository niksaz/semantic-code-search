import math
import tensorflow as tf


def init_net(nodes, children, feature_size, encoder_size):
    """Initialize an empty network."""

    with tf.name_scope('network'):
        # (batch_size x split_size x encoder_size)
        enc = encoder_layer(nodes, children, feature_size, encoder_size)

        # (batch_size x split_size x feature_size)
        gru_out = bigru_layer(enc, encoder_size, feature_size, 1)
        # (batch_size x feature_size x split_size)
        gru_out = tf.transpose(gru_out, perm=[0, 2, 1])

        with tf.name_scope('pooling'):
            # (batch_size x feature_size)
            hidden = tf.reduce_max(gru_out, axis=2)

    with tf.name_scope('summaries'):
        tf.summary.scalar('split_size', tf.shape(nodes)[1])
        tf.summary.scalar('tree_size', tf.shape(nodes)[2])
        tf.summary.scalar('child_size', tf.shape(children)[3])
        tf.summary.histogram('logits', hidden)
        tf.summary.image('inputs', tf.expand_dims(nodes, axis=4))
        tf.summary.image('enc', tf.expand_dims(enc, axis=4))
        tf.summary.image('bigru', tf.expand_dims(gru_out, axis=4))

    return hidden


def bigru_layer(enc, hidden_dim):
    gru_cell_fw = tf.nn.rnn_cell.GRUCell(hidden_dim // 2)
    gru_cell_bw = tf.nn.rnn_cell.GRUCell(hidden_dim // 2)

    # TODO: try dropout
    # gru_cell_fw = tf.nn.rnn_cell.DropoutWrapper(gru_cell_fw, output_keep_prob=0.05)
    # gru_cell_bw = tf.nn.rnn_cell.DropoutWrapper(gru_cell_bw, output_keep_prob=0.05)

    (gru_out_fw, gru_out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=gru_cell_fw,
        cell_bw=gru_cell_bw,
        inputs=enc,
        dtype=tf.float32)
    # (batch, split_size, hidden_dim)
    return tf.concat([gru_out_fw, gru_out_fw], axis=-1)


def encoder_layer(nodes, children, features_size, encoder_size):
    with tf.name_scope('encoder_layer'):
        with tf.name_scope('encoder_layer'):
            batch_size = tf.shape(nodes)[0]
            nodes_size = tf.shape(nodes)[2]
            children_size = tf.shape(children)[3]
            split_nodes = tf.reshape(nodes, (-1, nodes_size, features_size))
            split_children = tf.reshape(children, (-1, nodes_size, children_size))

        with tf.name_scope('encoder_mul'):
            node_list = []
            traverse_mul(split_nodes, split_children, node_list,
                         features_size, encoder_size)

        with tf.name_scope('encoder_pooling'):
            node_list = tf.stack(node_list)
            # ((batch_size * split_size) x encoder_size)
            enc_node = tf.reduce_max(node_list, axis=0)

        # (batch_size x split_size x encoder_size)
        return tf.reshape(enc_node, (batch_size, -1, encoder_size))


def traverse_mul(nodes, children, node_list, features_size, encoder_size):
    with tf.name_scope('split_encoder_node'):
        nodes_size = nodes.get_shape().as_list()[1]
        # (batch_size x num_nodes x encoder_size)
        encoded_nodes = W_c(nodes, features_size, encoder_size)
        for j in range(nodes_size - 1, -1, -1):
            # (batch_size x num_children x encoder_size)
            current_children = children_tensor(encoded_nodes, children, j, encoder_size)
            # (batch_size x  encoder_size)
            children_sum = tf.reduce_sum(current_children, axis=1, name='children_sum')
            # (batch_size x encoder_size
            current_nodes = encoded_nodes[:, j, :]
            # (batch_size x encoder_size)
            new_node_vector = current_nodes + children_sum

            encoded_nodes = tf.concat([
                encoded_nodes[:, :j, :],
                tf.expand_dims(new_node_vector, axis=1),
                encoded_nodes[:, j+1:, :]],
                axis=1)
            node_list.append(new_node_vector)


def children_tensor(nodes, children, num_node, encoder_size):
    with tf.name_scope('sum_children'):
        batch_size = nodes.get_shape().as_list()[0]
        zero_vecs = tf.zeros((batch_size, 1, encoder_size), tf.int32)
        # (batch_size x num_nodes x encoder_size)
        node_vectors = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # (batch_size x 2)
        children_indices = tf.concat([
            tf.reshape(tf.range(batch_size), (-1, 1)),
            tf.fill([batch_size, 1], num_node)],
            axis=1)
        # (batch_size x num_children)
        children_indices = tf.gather_nd(children, children_indices)
        children_size = tf.shape(children_indices)[1]
        # (batch_size x num_children x 1)
        children_indices = tf.reshape(children_indices, (batch_size, -1, 1))
        # (batch_size x num_children x 1)
        batch_indices = tf.tile(tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1)), [1, children_size, 1],
                                name='batch_indices').eval()
        # (batch_size x num_children x 2)
        children_indices = tf.concat([batch_indices, children_indices], axis=2, name='children_indices')
        # (batch_size x num_children x encoder_size)
        encoded_children = tf.gather_nd(node_vectors, children_indices, name='children_vectors')
        return encoded_children


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
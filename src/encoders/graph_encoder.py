import collections
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import tensorflow as tf
from dpu_utils.mlutils import Vocabulary

from utils import tfutils
from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import pool_sequence_embedding
from .encoder import Encoder, QueryType
from .utils import ggnn_network, tree_processing, great_transformer_network, rnn_network, util


class GraphEncoder(Encoder):
    encoder_hypers = {
        'token_vocab_size': 10000,
        'token_vocab_count_threshold': 10,
        'token_embedding_size': 128,
        'token_use_bpe': True,
        'token_pct_bpe': 0.5,
        'max_num_tokens': 200,
        'stack': [],
        'is_plain': False
    }

    @classmethod
    def update_config(cls, mode: str, is_plain: bool):
        cls.encoder_hypers['is_plain'] = is_plain
        if mode in ['ggnn', 'ggnnmodel']:
            cls.encoder_hypers['stack'] = ['ggnn-pure']
        elif mode in ['rnn-ggnn-sandwich']:
            cls.encoder_hypers['stack'] = ['rnn', 'ggnn', 'rnn', 'ggnn', 'rnn']
        elif mode in ['transformer-ggnn-sandwich']:
            cls.encoder_hypers['stack'] = ['transformer', 'ggnn', 'transformer', 'ggnn', 'transformer']
        elif mode in ['great', 'greatmodel']:
            cls.encoder_hypers['stack'] = ['great']
        elif mode in ['great10', 'great10model']:
            cls.encoder_hypers['stack'] = ['great']
            great_transformer_network.Transformer.default_config['num_layers'] = 10
        elif mode in ['transformer', 'transformermodel']:
            cls.encoder_hypers['stack'] = ['transformer']
        elif mode in ['transformer10', 'transformer10model']:
            cls.encoder_hypers['stack'] = ['transformer']
            great_transformer_network.Transformer.default_config['num_layers'] = 10
        elif mode in ['graphnbow', 'graphnbowmodel']:
            cls.encoder_hypers['stack'] = []
        elif mode in ['rnn', 'rnnmodel']:
            cls.encoder_hypers['stack'] = ['rnn']
        else:
            raise ValueError(f"Tried to update graph config with {mode}")

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        hypers = super().get_default_hyperparameters()
        hypers.update(cls.encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)
        assert label == 'code', 'GGNNEncoder should only be used for code'

    @property
    def output_representation_size(self) -> int:
        # assert self.get_hyper('type_embedding_size') == self.get_hyper('token_embedding_size')
        return self.get_hyper('token_embedding_size')

    def _make_placeholders(self):
        super()._make_placeholders()
        self.placeholders['node_masks'] = tf.placeholder(tf.float32, shape=[None, None], name='node_masks')
        self.placeholders['node_token_ids'] = tf.placeholder(tf.int32, shape=[None, None], name='node_token_ids')
        self.placeholders['edges'] = tf.placeholder(tf.int32, shape=[None, 4], name='edges')
        self.placeholders['seq_masks'] = tf.placeholder(tf.float32, shape=[None, None], name='seq_masks')
        self.placeholders['seq_token_ids'] = tf.placeholder(tf.int32, shape=[None, None], name='seq_token_ids')

    def token_embedding_layer(self, input_ids: tf.Tensor, suffix=None) -> tf.Tensor:
        token_embeddings = tf.get_variable(
            name=f'token_embeddings{suffix}',
            initializer=tf.glorot_uniform_initializer(),
            shape=[len(self.metadata['token_vocab']), self.get_hyper('token_embedding_size')]
        )
        self.__token_embeddings = token_embeddings

        token_embeddings = tf.nn.dropout(
            token_embeddings,
            keep_prob=self.placeholders['dropout_keep_rate']
        )
        return tf.nn.embedding_lookup(params=token_embeddings, ids=input_ids)

    def _build_stack(self, states, is_train: bool):
        stack = self.get_hyper('stack')
        if len(stack) == 0:
            return None
        if stack[0] != 'rnn':
            pos_enc = util.positional_encoding(self.get_hyper('token_embedding_size'), 5000)
            states += pos_enc[:tf.shape(states)[1]]

        vocab_dim = self.get_hyper(f'token_vocab_size')
        for kind in self.get_hyper('stack'):
            if kind in ['ggnn', 'ggnn-pure']:
                if kind == 'ggnn':
                    ggnn_network.GGNN.default_config['time_steps'] = [3, 1]
                elif kind == 'ggnn-pure':
                    ggnn_network.GGNN.default_config['time_steps'] = [3, 1, 3, 1]
                ggnn_model = ggnn_network.GGNN(vocab_dim=vocab_dim)
                states = ggnn_model(states, self.placeholders['edges'], is_train)
            elif kind == 'rnn':
                rnn_model = rnn_network.RNN(vocab_dim=vocab_dim)
                states = rnn_model(states, is_train)
            elif kind == 'great' or kind == 'transformer':
                config = great_transformer_network.Transformer.default_config
                if kind == 'transformer':
                    config['num_edge_types'] = None
                transformer_model = great_transformer_network.Transformer(config, vocab_dim=vocab_dim)

                mask = self.placeholders['node_masks']
                mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)
                edges = self.placeholders['edges']
                attention_bias = tf.stack([edges[:, 0], edges[:, 1], edges[:, 3], edges[:, 2]], axis=1)
                states = transformer_model(states, mask, attention_bias, training=is_train)

        return states

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope('graph_encoder'):
            self._make_placeholders()
            seq_tokens = self.token_embedding_layer(self.placeholders['seq_token_ids'], suffix='_seq')
            print('seq tokens', seq_tokens.shape)
            seq_token_masks = self.placeholders['seq_masks']
            print('seq token masks', seq_token_masks.shape)
            seq_token_lens = tf.reduce_sum(seq_token_masks, axis=1)  # B
            print('seq token lens', seq_token_lens.shape)
            token_encoding = pool_sequence_embedding('mean',
                                                     sequence_token_embeddings=seq_tokens,
                                                     sequence_lengths=seq_token_lens,
                                                     sequence_token_masks=seq_token_masks)
            print('token encoding', token_encoding.shape)

            node_tokens = self.token_embedding_layer(self.placeholders['node_token_ids'], suffix='_node')
            print('node tokens', node_tokens.shape)
            node_token_masks = self.placeholders['node_masks']
            print('node token masks', node_token_masks.shape)
            node_token_lens = tf.reduce_sum(node_token_masks, axis=1)  # B

            # node_encodings = self._build_stack(node_tokens, is_train)
            #
            # if node_encodings is not None:
            #     print('node encoding', node_encodings.shape)
            #     graph_encoding = pool_sequence_embedding('mean',
            #                                              sequence_token_embeddings=node_tokens,
            #                                              sequence_lengths=node_token_lens,
            #                                              sequence_token_masks=node_token_masks)

            node_encodings = self._build_stack(seq_tokens, is_train)

            if node_encodings is not None:
                print('node encoding', node_encodings.shape)
                graph_encoding = pool_sequence_embedding('mean',
                                                         sequence_token_embeddings=seq_tokens,
                                                         sequence_lengths=seq_token_lens,
                                                         sequence_token_masks=seq_token_masks)

        if node_encodings is None:
            return token_encoding
        if self.get_hyper('is_plain'):
            return graph_encoding

        return token_encoding + graph_encoding

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['token_counter'] = collections.Counter()
        raw_metadata['edge_types'] = set()
        raw_metadata['nodes_by_tokens'] = list()
        return raw_metadata

    @classmethod
    def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any], use_subtokens: bool = False,
                                  mark_subtoken_end: bool = False) -> None:
        hypers = cls.get_default_hyperparameters()
        assert 'nodes' in data_to_load
        assert 'edges' in data_to_load
        node_tokens = collections.Counter(token for token in data_to_load['nodes'])
        edge_types = {edge_type for edge_type in data_to_load['edges']}
        raw_metadata['token_counter'].update(node_tokens)
        raw_metadata['edge_types'] = raw_metadata['edge_types'].union(edge_types)
        raw_metadata['nodes_by_tokens'].append(len(data_to_load['sequence']))

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                          raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("Finalising metadata")
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = collections.Counter()
        merged_edge_types = set()
        token_counts = []
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']
            merged_edge_types = merged_edge_types.union(raw_metadata['edge_types'])
            token_counts.extend(raw_metadata['nodes_by_tokens'])

        if hyperparameters[f'{encoder_label}_token_use_bpe']:
            token_vocabulary = BpeVocabulary(
                vocab_size=hyperparameters[f'{encoder_label}_token_vocab_size'],
                pct_bpe=hyperparameters[f'{encoder_label}_token_pct_bpe']
            )
            token_vocabulary.fit(merged_token_counter)
            print('Total token word vocabulary words:', len(token_vocabulary.word_vocab))
            print('Total token bpe vocabulary words:', len(token_vocabulary.bpe_vocab))
        else:
            token_vocabulary = Vocabulary.create_vocabulary(
                tokens=merged_token_counter,
                max_size=hyperparameters[f'{encoder_label}_token_vocab_size'],
                count_threshold=hyperparameters[f'{encoder_label}_token_vocab_count_threshold'])
            print('Total token vocabulary words:', len(token_vocabulary.id_to_token))

        final_metadata['token_vocab'] = token_vocabulary
        final_metadata['edge_type_mapping'] = {edge_type: i for i, edge_type in enumerate(merged_edge_types)}
        print('Edge type mapping:', final_metadata['edge_type_mapping'])
        print("Percentiles:")
        for p in [0, 1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 99.9, 100]:
            print(p, np.percentile(token_counts, p))
        return final_metadata

    @classmethod
    def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                              data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:

        assert 'nodes' in data_to_load
        assert 'edges' in data_to_load

        node_tokens = data_to_load['nodes']
        seq_tokens = data_to_load['sequence']
        # print(seq_tokens)
        edges = np.array([
            (metadata['edge_type_mapping'][edge_type], v, u)
            for edge_type, edges_of_type in data_to_load['edges'].items()
            for v, u in edges_of_type
        ], dtype=np.int)

        node_token_ids, mask = (
            tfutils.convert_and_pad_token_sequence(
                metadata['token_vocab'],
                node_tokens,
                hyperparameters[f'{encoder_label}_max_num_tokens']
            )
        )
        seq_token_ids, seq_mask = (
            tfutils.convert_and_pad_token_sequence(
                metadata['token_vocab'],
                seq_tokens,
                hyperparameters[f'{encoder_label}_max_num_tokens']
            )
        )
        result_holder[f'{encoder_label}_node_masks'] = mask
        result_holder[f'{encoder_label}_node_token_ids'] = node_token_ids
        result_holder[f'{encoder_label}_edges'] = edges
        result_holder[f'{encoder_label}_seq_token_masks'] = seq_mask
        result_holder[f'{encoder_label}_seq_token_ids'] = seq_token_ids
        return True

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['node_masks'] = []
        batch_data['node_token_ids'] = []
        batch_data['edges'] = []
        batch_data['seq_masks'] = []
        batch_data['seq_token_ids'] = []

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:

        current_sample = {}
        current_sample['node_masks'] = sample[f'{self.label}_node_masks']
        current_sample['node_token_ids'] = sample[f'{self.label}_node_token_ids']
        current_sample['edges'] = sample[f'{self.label}_edges']
        current_sample['seq_masks'] = sample[f'{self.label}_seq_token_masks']
        current_sample['seq_token_ids'] = sample[f'{self.label}_seq_token_ids']

        for key, value in current_sample.items():
            if key in batch_data:
                batch_data[key].append(value)

        return False

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any],
                               is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        node_masks = batch_data['node_masks']
        node_token_ids = batch_data['node_token_ids']
        edges = batch_data['edges']
        seq_masks = batch_data['seq_masks']
        seq_token_ids = batch_data['seq_token_ids']

        if node_masks:
            # pad batches so that every batch has the same number of nodes
            max_tokens = max([len(x) for x in node_masks])
            node_masks = [list(n) + [0] * (max_tokens - len(n)) for n in node_masks]
            node_token_ids = [list(n) + [-1] * (max_tokens - len(n)) for n in node_token_ids]
            edges = [
                [edge_type, batch_id, v, u]
                for batch_id, edge_pack in enumerate(edges)
                for edge_type, v, u in edge_pack
                if v < max_tokens and u < max_tokens
            ]

        tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_masks'], node_masks)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_token_ids'], node_token_ids)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['edges'], edges)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['seq_masks'], seq_masks)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['seq_token_ids'], seq_token_ids)

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return self.__token_embeddings, list(self.metadata['token_vocab'].id_to_token)

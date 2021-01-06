import collections
from typing import Dict, Any, Tuple, List, Optional

import tensorflow as tf
from dpu_utils.mlutils import Vocabulary

from utils import tfutils
from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import pool_sequence_embedding
from .ast_encoder import _get_tree_elements_seq
from .encoder import Encoder, QueryType


class AstTokensEncoder(Encoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'token_vocab_size': 10000,
            'token_vocab_count_threshold': 10,
            'token_embedding_size': 128,
            'token_use_bpe': True,
            'token_pct_bpe': 0.5,

            'max_num_tokens': 100
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self) -> int:
        return self.get_hyper('token_embedding_size')

    def _make_placeholders(self):
        super()._make_placeholders()
        self.placeholders['node_masks'] = tf.placeholder(tf.float32, shape=[None, None], name='node_masks')
        self.placeholders['tokens'] = tf.placeholder(tf.int32, shape=[None, None], name='tokens')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope('ast_tokens_encoder'):
            self._make_placeholders()

            node_tokens = self.embedding_layer(self.placeholders['tokens'])
            node_token_masks = self.placeholders['node_masks']
            node_token_lens = tf.reduce_sum(node_token_masks, axis=1)  # B
            token_encoding = pool_sequence_embedding('mean',
                                                     sequence_token_embeddings=node_tokens,
                                                     sequence_lengths=node_token_lens,
                                                     sequence_token_masks=node_token_masks)
        return token_encoding

    def embedding_layer(self, input_ids: tf.Tensor) -> tf.Tensor:
        token_embeddings = tf.get_variable(
            name='token_embeddings',
            initializer=tf.glorot_uniform_initializer(),
            shape=[len(self.metadata['token_vocab']), self.get_hyper('token_embedding_size')])
        self.__embeddings = token_embeddings

        token_embeddings = tf.nn.dropout(
            token_embeddings,
            keep_prob=self.placeholders['dropout_keep_rate'])
        return tf.nn.embedding_lookup(params=token_embeddings, ids=input_ids)

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['token_counter'] = collections.Counter()
        return raw_metadata

    @classmethod
    def load_metadata_from_sample(cls, data_to_load: Any, raw_metadata: Dict[str, Any], use_subtokens: bool = False,
                                  mark_subtoken_end: bool = False) -> None:
        default_hypers = cls.get_default_hyperparameters()
        node_types, node_tokens = _get_tree_elements_seq(data_to_load, default_hypers['max_num_tokens'])
        raw_metadata['token_counter'].update(node_tokens)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                          raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = collections.Counter()
        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']
        if hyperparameters[f'{encoder_label}_token_use_bpe']:
            token_vocabulary = BpeVocabulary(
                vocab_size=hyperparameters[f'{encoder_label}_token_vocab_size'],
                pct_bpe=hyperparameters[f'{encoder_label}_token_pct_bpe'])
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
        return final_metadata

    @classmethod
    def load_data_from_sample(cls, encoder_label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any],
                              data_to_load: Any, function_name: Optional[str], result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:
        _, node_tokens = _get_tree_elements_seq(data_to_load, hyperparameters[f'{encoder_label}_max_num_nodes'])
        node_token_ids, mask = (
            tfutils.convert_and_pad_token_sequence(
                metadata['token_vocab'],
                node_tokens,
                hyperparameters[f'{encoder_label}_max_num_tokens']))
        result_holder[f'{encoder_label}_node_masks'] = list(mask)
        result_holder[f'{encoder_label}_tokens'] = list(node_token_ids)
        return True

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['node_masks'] = []
        batch_data['tokens'] = []

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool = False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        current_sample = {}
        current_sample['node_masks'] = sample[f'{self.label}_node_masks']
        current_sample['tokens'] = sample[f'{self.label}_tokens']
        for key, value in current_sample.items():
            if key in batch_data:
                batch_data[key].append(value)
        return False

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any],
                               is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        node_masks = batch_data['node_masks']
        node_token_ids = batch_data['tokens']
        if node_masks:
            max_tokens = max([len(x) for x in node_masks])
            node_masks = [n + [0] * (max_tokens - len(n)) for n in node_masks]
            node_token_ids = [n + [-1] * (max_tokens - len(n)) for n in node_token_ids]

        tfutils.write_to_feed_dict(feed_dict, self.placeholders['node_masks'], node_masks)
        tfutils.write_to_feed_dict(feed_dict, self.placeholders['tokens'], node_token_ids)

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return self.__embeddings, list(self.metadata['token_vocab'].id_to_token)

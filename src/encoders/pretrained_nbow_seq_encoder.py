# Author: Mikita Sazanovich

import pickle
from collections import Counter
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf

from dpu_utils.mlutils import Vocabulary
from utils.tfutils import pool_sequence_embedding
from .masked_seq_encoder import MaskedSeqEncoder
from utils import data_pipeline


class PretrainedNBoWEncoder(MaskedSeqEncoder):
    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def pretrained_embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
        resource = self.get_hyper('resource')
        embedding_path = f'resources/embeddings/{resource}/embeddings.npy'
        embedding = np.load(embedding_path)
        N, K = embedding.shape
        # Add a zero embedding for the UNK token.
        embedding = np.vstack([np.zeros((1, K), dtype=embedding.dtype), embedding])
        print('EMBEDDING', embedding.shape)
        token_embeddings = tf.get_variable(
            name="token_embeddings",
            shape=embedding.shape,
            initializer=tf.constant_initializer(embedding),
            trainable=True)  # trainable=False
        self.__embeddings = token_embeddings
        token_embeddings = tf.nn.dropout(token_embeddings, keep_prob=self.placeholders['dropout_keep_rate'])
        return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any],
                          raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        hypers = cls.get_default_hyperparameters()
        resource = hypers['resource']
        vocabulary_path = f'resources/embeddings/{resource}/token_to_index.pickle'
        with open(vocabulary_path, 'rb') as fin:
            token_to_index = pickle.load(fin)
        # Fictive counts so that the ordering in the internal vocabulary will be the same as the indices in the dict.
        token_to_count = {}
        for token, index in token_to_index.items():
            token_to_count[token] = len(token_to_index) - index
        token_counter = Counter(token_to_count)
        token_vocabulary = Vocabulary.create_vocabulary(
            tokens=token_counter,
            max_size=hyperparameters['%s_token_vocab_size' % encoder_label],
            count_threshold=0)
        print('token_to_index', token_to_index)
        print('token_vocabulary.id_to_token', token_vocabulary.id_to_token)

        final_metadata = {}
        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = token_counter.most_common(50)
        return final_metadata

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("nbow_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.pretrained_embedding_layer(self.placeholders['tokens'])
            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
            return pool_sequence_embedding(
                self.get_hyper('nbow_pool_mode').lower(),
                sequence_token_embeddings=seq_tokens_embeddings,
                sequence_lengths=seq_token_lengths,
                sequence_token_masks=seq_token_mask)


class ASTPretrainedNBoWEncoder(PretrainedNBoWEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'nbow_pool_mode': 'weighted_mean',
            'resource': data_pipeline.TREE_LABEL,
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers


class GraphPretrainedNBoWEncoder(PretrainedNBoWEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'nbow_pool_mode': 'weighted_mean',
            'resource': data_pipeline.GRAPH_LABEL,
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

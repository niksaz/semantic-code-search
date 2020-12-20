# Author: Mikita Sazanovich

from typing import Dict, Any

import numpy as np
import tensorflow as tf

from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding


class PretrainedNBoWEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {'nbow_pool_mode': 'weighted_mean'}
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def pretrained_embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
        resource = '_graphs'
        embedding_path = f'/home/zerogerc/msazanovich/CodeSearchNet/resources/embeddings/{resource}/embeddings.npy'
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

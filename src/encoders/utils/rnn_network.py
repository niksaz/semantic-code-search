"""
Code was copied from https://github.com/VHellendoorn/ICLR20-Great
Author: https://github.com/VHellendoorn

MIT License

Copyright (c) 2020 Vincent Hellendoorn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf


class RNN(tf.keras.layers.Layer):

	default_config = {
		'hidden_dim': 128,
		'dropout_rate': 0.1,
		'num_edge_types': 9,
		'num_layers': 2
	}

	def __init__(self, model_config=None, shared_embedding=None, vocab_dim=None):
		super(RNN, self).__init__()
		if model_config is None:
			model_config = RNN.default_config
		self.hidden_dim = model_config['hidden_dim']
		self.num_layers = model_config['num_layers']
		self.dropout_rate = model_config['dropout_rate']
		
		# Initialize embedding variable in constructor to allow reuse by other models
		if shared_embedding is not None:
			self.embed = shared_embedding
		elif vocab_dim is None:
			raise ValueError('Pass either a vocabulary dimension or an embedding Variable')
		else:
			random_init = tf.random_normal_initializer(stddev=self.hidden_dim ** -0.5)
			self.embed = tf.Variable(random_init([vocab_dim, self.hidden_dim]), dtype=tf.float32)
	
	def build(self, _):
		self.rnns_fwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True) for _ in range(self.num_layers)]
		self.rnns_bwd = [tf.keras.layers.GRU(self.hidden_dim//2, return_sequences=True, go_backwards=True) for _ in range(self.num_layers)]
	
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.bool)])
	def call(self, states, training):
		states = tf.ensure_shape(states, (None, None, self.hidden_dim))
		# Run states through all layers.
		real_dropout_rate = self.dropout_rate * tf.cast(training, 'float32')  # Easier for distributed training than an explicit conditional
		for layer_no in range(self.num_layers):
			fwd = self.rnns_fwd[layer_no](states)
			bwd = self.rnns_bwd[layer_no](states)
			states = tf.concat([fwd, bwd], axis=-1)			
			states = tf.nn.dropout(states, rate=real_dropout_rate)
		return states
	
	# Embed inputs. Note, does not add positional encoding.
	@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)])
	def embed_inputs(self, inputs):
		states = tf.nn.embedding_lookup(self.embed, inputs)
		states *= tf.math.sqrt(tf.cast(tf.shape(states)[-1], 'float32'))
		return states
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

import numpy as np
import tensorflow as tf

# Based on https://github.com/DongjunLee/transformer-tensorflow/blob/master/transformer/attention.py
def positional_encoding(dim, sentence_length, dtype=tf.float32):
	encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
	encoded_vec[::2] = np.sin(encoded_vec[::2])
	encoded_vec[1::2] = np.cos(encoded_vec[1::2])
	return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def prefix_sum(arr):
	res = [0]
	for a in arr: res.append(res[-1] + a)
	return res

# flowtron.py


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def get_gate_mask_from_lengths(lengths):
	max_len = tf.math.reduce_max(lengths).numpy().item()
	#ids = tf.range(0, max_len, dtype=tf.int64)
	#mask = ids < tf.expand_dims(lengths, 1)
	mask = tf.sequence_mask(lengths, max_len, dtype=tf.bool)
	return mask


def get_mask_from_lengths(lengths):
	max_len = tf.math.reduce_max(lengths).numpy().item()
	mask = tf.sequence_mask(lengths, max_len, dtype=tf.bool)
	return mask


class LinearNorm(layers.Layer):
	def __init__(self, dims, use_bias=True):
		super(LinearNorm, self).__init__()
		self.linear_layer = layers.Dense(
			dims, activation=None, use_bias=use_bias
		)


	def call(self, inputs):
		return self.linear_layer(inputs)


class ConvNorm(layers.Layer):
	def __init__(self, filters, kernel_size=1, strides=1, padding=None, 
			dilation=1, use_bias=True):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert kernel_size % 2 == 1
			padding = int(dilation * (kernel_size - 1) / 2)
		padding = "valid" if padding == 0 else "same"

		self.conv = layers.Conv1D(
			filters, kernel_size=kernel_size, strides=strides,
			padding=padding, dilation_rate=dilation
		)


	def call(self, inputs):
		return self.conv(inputs)


class GaussianMixture(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class MelEncoder(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class DenseLayer(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class Encoder(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class Attention(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class AR_Back_Step(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class AR_Step(layers.Layer):
	def __init__(self,):
		pass


	def call(self, inputs):
		pass


class Flowtron(Model):
	def __init__(self,):
		pass



	def call(self, inputs):
		pass


	def train_step(self, data):
		pass
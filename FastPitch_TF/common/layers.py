# layers.py


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LinearNorm(layers.Layer):
	def __init__(self, dims, use_bias=True, activation=None):
		super(LinearNorm, self).__init__()
		self.linear_layer = layers.Dense(
			dims, activation=activation, use_bias=use_bias
		)


	def call(self, inputs):
		return self.linear_layer(inputs)


class ConvNorm(layers.Layer):
	def __init__(self, filters, kernel_size=1, strides=1, padding=None, 
			dilation=1, use_bias=True, batch_norm=False):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert kernel_size % 2 == 1
			padding = int(dilation * (kernel_size - 1) / 2)
		# padding = "valid" if padding == 0 else "same"
		padding = "same" if padding else "causal"

		self.conv = layers.Conv1D(
			filters, kernel_size=kernel_size, strides=strides,
			padding=padding, dilation_rate=dilation, 
			activation=activation
		)
		self.norm = layers.BatchNormalization if batch_norm else None


	def call(self, inputs):
		if self.norm is None:
			return self.conv(inputs)
		else:
			return self.norm(self.conv(inputs))


class ConvReLUNorm(layers.Layer):
	def __init__(self, dims, kernel_size=1, 
			dropout=0.0):
		super(ConvReLUNorm, self).__init__()
		self.relu = layers.ReLU()
		# padding = "valid" if padding == 0 else "same"
		padding = "same" if kernel_size // 2 else "causal"
		self.conv = layers.Conv1D(
			dim, kernel_size=kernel_size, padding=padding
		)
		self.norm = layers.LayerNormalization()
		self.dropout = layers.Dropout(dropout)


	def call(self, signal):
		out = self.relu(self.conv(signal))
		out = self.norm(out)
		return self.dropout(out)
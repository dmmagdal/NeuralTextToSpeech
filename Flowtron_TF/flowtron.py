# flowtron.py


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
import tensorflow_addons as tfa


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
	def __init__(self, dims, use_bias=True, activation=None):
		super(LinearNorm, self).__init__()
		self.linear_layer = layers.Dense(
			dims, activation=activation, use_bias=use_bias
		)


	def call(self, inputs):
		return self.linear_layer(inputs)


class ConvNorm(layers.Layer):
	def __init__(self, filters, kernel_size=1, strides=1, padding=None, 
			dilation=1, use_bias=True, activation=None):
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


	def call(self, inputs):
		return self.conv(inputs)


class AttentionConditioningLayer(layers.Layer):
	def __init__(self, input_dim=2, attention_n_filters=32, 
			attention_kernel_sizes=[5, 3], attention_dim=64):
		super(AttentionConditioningLayer, self).__init__()
		self.location_connv_hidden = ConvNorm(
			attention_n_filters, kernel_size=attention_kernel_sizes[0],
			padding=None, use_bias=True, strides=1, dilation=1,
			activation="relu"
		)
		self.location_conv_out = ConvNorm(
			attention_dim, kernel_size=attention_kernel_sizes[3],
			padding=None, use_bias=True, strides=1, dilation=1,
			activation="sigmoid"
		)
		self.conv_layers = Sequential([
			self.location_connv_hidden,
			layers.ReLU(),
			self.location_conv_out,
			layers.Activation("sigmoid")
		])


	def call(self, attention_weights_cat):
		return self.conv_layers(attention_weights_cat)


class Attention(layers.Layer):
	def __init__(self, n_mel_channels=80, n_speaker_dim=128,
			n_text_channels=512, n_attn_channels=128, temperature=1.0):
		pass


	def call(self, queries, keys, values, mask=None, attn=None, 
			attn_prior=None):
		pass


class GaussianMixture(layers.Layer):
	def __init__(self, n_hidden, n_components, n_mel_channels, 
			fixed_gaussian):
		super(GaussianMixture, self).__init__()
		self.n_mel_channels = n_mel_channels
		self.n_components = n_components
		self.fixed_gaussian = fixed_gaussian
		self.mean_scale = mean_scale

		self.prob_layer = LinearNorm(n_components)

		if not fixed_gaussian:
			self.mean_layer = LinearNorm(n_mel_channels * n_components)
			self.log_var_layer = LinearNorm(n_components)
		else:
			mean = self.generate_mean(
				n_mel_channels, n_components, mean_scale
			)
			log_var = self.generate_log_var(
				n_mel_channels, n_components
			)
			self.mean = tf.cast(mean, dtype=tf.float32)
			self.log_var = tf.cast(log_var, dtype=tf.float32)


	def generate_mean(self, n_dimensions, n_components, scale=3):
		means = tf.cast(tf.eye(n_dimensions), dtype=tf.float32)
		ids = np.random.choice(
			range(n_dimensions), n_components, replace=False
		)
		means = means[ids] * scale
		means = tf.transpose(means, [1, 0])
		means = means[None]
		return means


	def generate_log_var(self, n_dimensions, n_components):
		log_var = tf.zeros(
			(1, n_dimensions, n_components), dtype=tf.float32
		)
		return log_var


	def generate_prob(self):
		return tf.ones((1, 1), dtype=tf.float32)


	def call(self, inputs, bs):
		prob = tf.math.softmax(self.prob_layer(inputs), axis=1)

		if not self.fixed_gaussian:
			mean = tf.reshape(
				self.mean_layer(inputs), 
				[bs, self.n_mel_channels, self.n_components]
			)
			log_var = tf.reshape(
				self.log_var_layer(inputs),
				[bs, self.n_mel_channels, self.n_components]
			)
		else:
			mean = self.mean
			log_var = self.log_var

		return mean, log_var, prob


class MelEncoder(layers.Layer):
	def __init__(self, encoder_embedding_dim=512, 
			encoder_kernel_size=3, encoder_n_convolutions=2,
			norm_fn=tfa.layers.InstanceNormalization()):
		super(MelEncoder, self).__init__()

		convolutions = []
		for i in range(encoder_n_convolutions):
			conv_layer = Sequential([
				ConvNorm(
					encoder_embedding_dim, 
					kernel_size=encoder_kernel_size, strides=1,
					padding=int((encoder_kernel_size - 1) / 2), 
					dilation=1, activation="relu"
				),
				norm_fn,
				layers.ReLU(),
				layers.Dropout(0.5)
			])
			convolutions.append(conv_layer)
		self.convolutions = Sequential(convolutions)

		self.lstm = layers.Bidirectional(
			layers.LSTM(int(encoder_embedding_dim / 2)),
			return_sequences=True
		)


	def call(self, inputs, lens):
		'''
		if tf.shape(inputs)[0] > 1:
			x_embedded = []

			for b_ind in range(tf.shape(inputs)[0]):
				curr_x = tf.identity(
					inputs[b_ind:b_ind + 1, :, :lens[b_ind]]
				)
				curr_x = self.convolutions(curr_x)
				x_embedded.append(curr_x[0])
		else:
			inputs = self.convolutions(inputs)
		'''

		inputs = self.convolutions(inputs)

		x, _ = self.lstm(inputs)

		x = tf.math.reduce_mean(x, axis=0)
		return x


class DenseLayer(layers.Layer):
	def __init__(self, sizes=[1024, 1024]):
		super(DenseLayer, self).__init__()
		dense_layers = []
		for size in sizes:
			dense_layers += [
				LinearNorm(size, use_bias=True),
				layers.Activation("tanh")
			]
		self.layers = Sequential(dense_layers)


	def call(self, inputs):
		return self.layers(inputs)
		

class Encoder(layers.Layer):
	def __init__(self,encoder_n_convolutions=3, 
			encoder_embedding_dim=512, encoder_kernel_size=5, 
			norm_fn=tfa.layers.InstanceNormalization()):
		super(Encoder, self).__init__()

		convolutions = []
		for i in range(encoder_n_convolutions):
			conv_layer = Sequential([
				ConvNorm(
					encoder_embedding_dim, 
					kernel_size=encoder_kernel_size, strides=1,
					padding=int((encoder_kernel_size - 1) / 2), 
					dilation=1, activation="relu"
				),
				norm_fn,
				layers.ReLU(),
				layers.Dropout(0.5)
			])
			convolutions.append(conv_layer)
		self.convolutions = Sequential(convolutions)

		self.lstm = layers.Bidirectional(
			layers.LSTM(int(encoder_embedding_dim / 2)),
			return_sequences=True
		)


	def call(self, inputs, lens):
		'''
		if tf.shape(inputs)[0] > 1:
			x_embedded = []

			for b_ind in range(tf.shape(inputs)[0]):
				curr_x = tf.identity(
					inputs[b_ind:b_ind + 1, :, :lens[b_ind]]
				)
				curr_x = self.convolutions(curr_x)
				x_embedded.append(curr_x[0])
		else:
			inputs = self.convolutions(inputs)
		'''

		inputs = self.convolutions(inputs)

		x, _ = self.lstm(inputs)

		x = tf.math.reduce_mean(x, axis=0)
		return x


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
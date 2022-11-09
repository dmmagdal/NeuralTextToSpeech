""" from https://github.com/jaywalnut310/glow-tts """
# text_encoder.py


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.utils import sequence_mask, convert_pad_shape


class LayerNorm(layers.Layer):
	def __init__(self, channels, eps=1e-4):
		super(LayerNorm, self).__init__()
		self.channels = channels
		self.eps = eps

		# self.gamma = tf.ones(channels)
		# self.beta = tf.zeros(channels)
		self.gamma = self.add_weight(
			name="gamma",
			shape=[channels],
			initializer="ones"
		)
		self.beta = self.add_weight(
			name="beta",
			shape=[channels],
			initializer="zeros"
		)


	def call(self, x):
		n_dims = len(x.shape)
		mean = tf.math.reduce_mean(x, 1, keep_dims=True)
		variance = tf.math.reduce_mean(
			(x - mean) ** 2, 2, keep_dims=True
		)

		x = (x - mean) * tf.math.rsqrt(variance + self.eps)

		shape = [1, -1] + [1] * (n_dims - 2)
		x = x * tf.reshape(self.gamma, *shape) +\
			tf.reshape(self.beta, *shape)
		return x


class ConvReluNorm(layers.Layer):
	def __init__(self, hidden_channels, out_channels, kernel_size,
			n_layers, p_dropout):
		super(ConvReluNorm, self).__init__()
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
		padding = "causal" if kernel_size // 2 else "same"

		self.conv_layers = []
		self.norm_layers = []
		self.conv_layers.append(
			layers.Conv1D(
				hidden_channels, kernel_size=kernel_size,
				padding=padding
			)
		)
		self.norm_layers.append(LayerNorm(hidden_channels))
		self.relu_drop = keras.Sequential([
			layers.ReLU(),
			layers.Dropout(p_dropout),
		])
		for _ in range(n_layers - 1):
			self.conv_layers.append(
				layers.Conv1D(
					hidden_channels, kernel_size=kernel_size,
					padding=padding
				)
			)
			self.norm_layers.append(LayerNorm(hidden_channels))
		self.proj = layers.Conv1D(out_channels, 1)


	def call(self, x, x_mask):
		x_org = x
		for i in range(self.n_layers):
			x = self.conv_layers[i](x * x_mask)
			x = self.norm_layers[i](x)
			x - self.relu_drop(x)
		x = x_org + self.proj(x)
		return x * x_mask


class DurationPredictor(layer.Layer):
	def __init__(self, filter_channels, kernel_size, p_dropout):
		super(DurationPredictor, self).__init__()
		self.filter_channels = filter_channels
		self.p_dropout = p_dropout
		padding = "causal" if kernel_size // 2 else "same"

		self.drop = layers.Dropout(p_dropout)
		self.conv1 = layers.Conv1D(
			filter_channels, kernel_size=kernel_size, padding=padding
		)
		self.norm1 = LayerNorm(filter_channels)
		self.conv2 = layers.Conv1D(
			filter_channels, kernel_size=kernel_size, padding=padding
		)
		self.norm2 = LayerNorm(filter_channels)
		self.proj = layers.Conv1D(1, 1)


	def call(self, x, x_mask):
		x = self.conv1(x * x_mask)
		x = tf.nn.relu(x)
		x = self.norm1(x)
		x = self.drop(x)
		x = self.conv2(x * x_mask)
		x = tf.nn.relu(x)
		x = self.norm2(x)
		x = self.drop(x)
		x = self.proj(x * x_mask)
		return x * x_mask


class MultiHeadAttention(layers.Layer):
	def __init__(self, filter_channels, out_channels, n_heads, 
			window_size=None, heads_share=True, p_dropout=0.0,
			proximal_bias=False, proximal_init=False):
		super(MultiHeadAttention, self).__init__()
		assert channels % n_heads == 0

		self.channels = channels
		self.out_channels = out_channels
		self.n_heads = n_heads
		self.window_size = window_size
		self.heads_share = heads_share
		self.proximal_bias = proximal_bias
		self.p_dropout = p_dropout
		self.attn = None

		self.k_channels = channels // n_heads
		self.conv_q = layers.Conv1D(channels, 1)
		self.conv_k = layers.Conv1D(channels, 1)
		self.conv_v = layers.Conv1D(channels, 1)
		if window_size is not None:
			n_heads_rel = 1 if heads_share else n_heads
			rel_stddev = self.k_channels ** -0.5
			self.emb_rel_k = self.add_weight()
			self.emb_rel_v = self.add_weight()
		self.conv_o = layers.Conv1D(out_channels, 1)
		self.drop = layers.Dropout(p_dropout)

		if proximal_init:
			self.conv_k = self.conv_q


	def call(self, x, c, attn_mask=None):
		q = self.conv_q(x)
		k = self.conv_q(c)
		v = self.conv_v(c)

		x, self.attn = self.attention(q, k, v, mask=attn_mask)

		x = self.conv_o(x)
		return x


	def attention(self, query, key, value, mask=None):
		b, d, t_s, t_t = (*key.shape, query.shape[2])
		query = tf.transpose(tf.reshape(
			query, [b, self.n_heads, self.k_channels, t_t]
		), [0, 1, 3, 2])
		key = tf.transpose(tf.reshape(
			key, [b, self.n_heads, self.k_channels, t_s]
		), [0, 1, 3, 2])
		value = tf.transpose(tf.reshape(
			value, [b, self.n_heads, self.k_channels, t_s]
		), [0, 1, 3, 2])

		scores = tf.linalg.matmul(query, tf.transpose(value, []))


class FFN(layers.Layer):
	def __init__(self, out_channels, filter_channels, kernel_size,
			p_dropout=0.0):
		super(FFN, self).__init__()
		self.out_channels = out_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		padding = "causal" if padding else "same"

		self.conv1 = layers.Conv1D(
			filter_channels, kernel_size=kernel_size,
			padding=padding
		)
		self.conv2 = layers.Conv1D(
			out_channels, kernel_size=kernel_size,
			padding=padding
		)
		self.relu = layers.ReLU()
		self.drop = layers.Dropout(p_dropout)


	def call(self, x, x_mask):
		x = self.conv1(x * x_mask)
		x = self.relu(x)
		x = self.drop(x)
		x = self.conv2(x * x_mask)
		return x * x_mask


class Encoder(layers.Layer):
	def __init__(self, hidden_channels, filter_channels, n_heads,
			n_layers, kernel_size=1, p_dropout=0.0, window_size=None,
			**kwargs):
		super(Encoder, self).__init__()
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.window_size = window_size

		self.drop = layers.Dropout(p_dropout)
		self.attn_layers = []
		self.norm_layers1 = []
		self.ffn_layers = []
		self.norm_layers2 = []
		for _ in range(self.n_layers):
			self.attn_layers.append(
				
			)
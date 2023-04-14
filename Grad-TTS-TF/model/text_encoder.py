""" from https://github.com/jaywalnut310/glow-tts """
# text_encoder.py


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.utils import sequence_mask, convert_to_shape#convert_pad_shape


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
			initializer="ones",
			trainable=True,
		)
		self.beta = self.add_weight(
			name="beta",
			shape=[channels],
			initializer="zeros",
			trainable=True,
		)
		self.layernorm = layers.LayerNormalization(epsilon=self.eps)


	def call(self, x):
		'''
		print(f"layer norm x: {x}, {x.shape}")
		n_dims = len(x.shape)
		print(f"ndims: {n_dims}")
		mean = tf.math.reduce_mean(x, 1, keepdims=True)
		print(f"mean: {mean}, {mean.shape}")
		variance = tf.math.reduce_mean(
			(x - mean) ** 2, 1, keepdims=True
		)
		print(f"variance: {variance}, {variance.shape}")

		x = (x - mean) * tf.math.rsqrt(variance + self.eps)
		print(f"x: {x}, {x.shape}")

		shape = [1, -1] + [1] * (n_dims - 2)
		print(f"shape: {shape}")
		# x = x * tf.reshape(self.gamma, *shape) +\
		# 	tf.reshape(self.beta, *shape)
		x = x * tf.identity(self.gamma).reshape(*shape) +\
			tf.identity(self.beta).reshape(*shape)
		print(f"x: {x}, {x.shape}")
		return x
		'''
		return self.layernorm(x)


class ConvReluNorm(layers.Layer):
	def __init__(self, hidden_channels, out_channels, kernel_size,
			n_layers, p_dropout):
		super(ConvReluNorm, self).__init__()
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout
		padding = "causal" if kernel_size // 2 else "same" # Not sure which to use
		# padding = "same" if kernel_size // 2 else "valid"

		self.conv_layers = []
		self.norm_layers = []
		self.conv_layers.append(
			layers.Conv1D(
				# hidden_channels, kernel_size=kernel_size,
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
		self.proj = layers.Conv1D(out_channels, kernel_size=1)
		# self.proj.build(self.hidden_channels)
		# self.proj.weight.assign(tf.zeros_like(self.proj.weight))
		# self.proj.bias.assign(tf.zeros_like(self.proj.bias))


	def call(self, x, x_mask):
		x_org = x
		for i in range(self.n_layers):
			x = self.conv_layers[i](x * x_mask)
			x = self.norm_layers[i](x)
			x - self.relu_drop(x)
		x = x_org + self.proj(x)
		return x * x_mask


class DurationPredictor(layers.Layer):
	def __init__(self, filter_channels, kernel_size, p_dropout):
		super(DurationPredictor, self).__init__()
		self.filter_channels = filter_channels
		self.p_dropout = p_dropout
		padding = "causal" if kernel_size // 2 else "same" # Not sure which one to use.
		# padding = "same" if kernel_size // 2 else "valid"

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
	def __init__(self, channels, out_channels, n_heads, 
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
		self.conv_q = layers.Conv1D(
			channels, 1, 
			kernel_initializer=keras.initializers.GlorotUniform()
		)
		self.conv_k = layers.Conv1D(
			channels, 1, 
			kernel_initializer=keras.initializers.GlorotUniform()
		)
		self.conv_v = layers.Conv1D(
			channels, 1, 
			kernel_initializer=keras.initializers.GlorotUniform()
		)
		if window_size is not None:
			n_heads_rel = 1 if heads_share else n_heads
			rel_stddev = self.k_channels ** -0.5
			self.emb_rel_k = tf.Variable(
				initial_value=tf.random.normal(
					[n_heads_rel, window_size * 2 + 1, self.k_channels],
					stddev=rel_stddev
				)
			)
			self.emb_rel_v = tf.Variable(
				initial_value=tf.random.normal(
					[n_heads_rel, window_size * 2 + 1, self.k_channels],
					stddev=rel_stddev
				)
			)
		self.conv_o = layers.Conv1D(out_channels, 1)
		self.drop = layers.Dropout(p_dropout)

		if proximal_init: # False by default and never altered.
			self.conv_k = self.conv_q


	def call(self, x, c, attn_mask=None):
		q = self.conv_q(x)
		k = self.conv_k(c)
		v = self.conv_v(c)

		x, self.attn = self.attention(q, k, v, mask=attn_mask)

		x = self.conv_o(x)
		return x


	def attention(self, query, key, value, mask=None):
		# b, d, t_s, t_t = (*key.shape, query.shape[2])
		b, t_s, d, t_t = (*key.shape, query.shape[1])
		query = tf.transpose(tf.reshape(
			query, [b, self.n_heads, self.k_channels, t_t]
		), [0, 1, 3, 2])
		key = tf.transpose(tf.reshape(
			key, [b, self.n_heads, self.k_channels, t_s]
		), [0, 1, 3, 2])
		value = tf.transpose(tf.reshape(
			value, [b, self.n_heads, self.k_channels, t_s]
		), [0, 1, 3, 2])

		scores = tf.linalg.matmul(
			query, tf.transpose(key, [0, 1, 3, 2])
		) / math.sqrt(self.k_channels)
		if self.window_size is not None:
			assert t_s == t_t, "Relative attention is only available for self-attention."
			key_relative_embeddings = self._get_relative_embeddings(
				self.emb_rel_k, t_s
			)
			rel_logits = self._matmul_with_relative_keys(
				query, key_relative_embeddings
			)
			rel_logits = self._relative_position_to_absolute_position(
				rel_logits
			)
			scores_local = rel_logits / math.sqrt(self.k_channels)
			scores = scores + scores_local
		if self.proximal_bias: # Default is False and never set so I dont really know why they kept/put this in.
			assert t_s == t_t, "Proximal bias is only available for self-attention."
			scores = scores + tf.cast(
				self._attention_bias_proximal(t_s), 
				dtype=scores.dtype
			)

		if mask is not None:
			# scores = tf.tile(scores, mask == 0, -1e4)
			scores = tf.where(tf.equal(scores, 0), tf.ones_like(scores) * -1e4, scores)
		p_attn = tf.nn.softmax(scores, axis=-1)
		p_attn = self.drop(p_attn)
		output = tf.linalg.matmul(p_attn, value)
		if self.window_size is not None:
			relative_weights = self._absolute_position_to_relative_position(
				p_attn
			)
			value_relative_embeddings = self._get_relative_embeddings(
				self.emb_rel_v, t_s
			)
			output = output + self._matmul_with_relative_values(
				relative_weights, value_relative_embeddings
			)
		output = tf.reshape(tf.transpose(output, [0, 1, 3, 2]), [b, t_t, d])
		return output, p_attn


	def _matmul_with_relative_values(self, x, y):
		ret = tf.linalg.matmul(x, tf.expand_dims(y, 0))
		return ret


	def _matmul_with_relative_keys(self, x, y):
		ret = tf.linalg.matmul(
			x, tf.transpose(tf.expand_dims(y, 0), [0, 1, 3, 2])
		)
		return ret


	def _get_relative_embeddings(self, relative_embeddings, length):
		pad_length = max(length - (self.window_size + 1), 0)
		slice_start_position = max((self.window_size + 1) - length, 0)
		slice_end_position = slice_start_position + 2 * length - 1
		if pad_length > 0:
			pad = convert_to_shape(
				[[0, 0], [pad_length, pad_length], [0, 0]]
			)
			padded_relative_embeddings = tf.pad(
				# relative_embeddings, convert_to_shape(
				# 	[[0, 0], [pad_length, pad_length], [0, 0]]
				# )
				relative_embeddings,
				[[0, 0], [pad_length, pad_length], [0, 0]]
			)
		else:
			padded_relative_embeddings = relative_embeddings
		used_relative_embeddings = padded_relative_embeddings[
			:, slice_start_position:slice_end_position
		]
		return used_relative_embeddings


	def _relative_position_to_absolute_position(self, x):
		batch, heads, length, _ = x.shape
		x = tf.pad(
			# x, convert_to_shape([[0, 0], [0, 0], [0, 0], [0, 1]])
			x, [[0, 0], [0, 0], [0, 0], [0, 1]]
		)
		x_flat = tf.reshape(x, [batch, heads, length * 2 * length])
		x_flat = tf.pad(
			# x_flat, convert_to_shape([[0, 0], [0, 0], [0, length - 1]])
			x_flat, [[0, 0], [0, 0], [0, length - 1]]
		)
		x_final = tf.reshape(
			x_flat, [batch, heads, length + 1, 2 * length - 1]
		)[:, :, :length, length - 1:]
		return x_final


	def _absolute_position_to_relative_position(self, x):
		batch, heads, length, _ = x.shape
		x = tf.pad(
			# x, convert_to_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
			x, [[0, 0], [0, 0], [0, 0], [0, length - 1]]
		)
		x_flat = tf.reshape(
			x, [batch, heads, length ** 2 + length * (length - 1)]
		)
		x_flat = tf.pad(
			# x_flat, convert_to_shape([[0, 0], [0, 0], [length, 0]])
			x_flat, [[0, 0], [0, 0], [length, 0]]
		)
		x_final = tf.reshape(
			x_flat, [batch, heads, length, length * 2]
		)[:, :, :, 1:]
		return x_final


	def _attention_bias_proximal(self, length):
		r = tf.random.normal(length, dtype=tf.float32)
		diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
		return tf.expand_dims(
			tf.expand_dims(-tf.math.log1p(tf.math.abs(diff)), 0), 0
		)


class FFN(layers.Layer):
	def __init__(self, out_channels, filter_channels, kernel_size,
			p_dropout=0.0):
		super(FFN, self).__init__()
		self.out_channels = out_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		padding = "causal" if (kernel_size // 2) else "same"

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
				MultiHeadAttention(
					hidden_channels, hidden_channels, n_heads, 
					window_size=window_size, p_dropout=p_dropout
				)
			)
			self.norm_layers1.append(LayerNorm(hidden_channels))
			self.ffn_layers.append(
				FFN(
					hidden_channels, filter_channels, kernel_size,
					p_dropout=p_dropout
				)
			)
			self.norm_layers2.append(LayerNorm(hidden_channels))


	def call(self, x, mask):
		# attn_mask = tf.expand_dims(mask, 2) * tf.expand_dims(mask, -1)
		attn_mask = tf.expand_dims(mask, 1) * tf.expand_dims(mask, -2)
		for i in range(self.n_layers):
			x = x * mask
			y = self.attn_layers[i](x, x, attn_mask)
			y = self.drop(y)
			x = self.norm_layers1[i](x + y)
			y = self.ffn_layers[i](x, mask)
			y = self.drop(y)
			x = self.norm_layers2[i](x + y)
		x = x * mask
		return x


class TextEncoder(keras.Model):
	def __init__(self, n_vocab, n_feats, n_channels, filter_channels,
			filter_channels_dp, n_heads, n_layers, kernel_size,
			p_dropout, window_size=None, spk_emb_dim=64, n_spks=1):
		super(TextEncoder, self).__init__()
		self.n_vocab = n_vocab
		self.n_feats = n_feats
		self.n_channels = n_channels
		self.filter_channels = filter_channels
		self.filter_channels_dp = filter_channels_dp
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.window_size = window_size
		self.spk_emb_dim = spk_emb_dim
		self.n_spks = n_spks

		self.emb = layers.Embedding(
			n_vocab, n_channels, 
			embeddings_initializer=keras.initializers.RandomNormal(
				mean=0.0, stddev=n_channels**-0.5
			)
		)

		self.prenet = ConvReluNorm(
			n_channels, n_channels, kernel_size=5, n_layers=3, 
			p_dropout=0.5
		)

		self.encoder = Encoder(
			n_channels + (spk_emb_dim if n_spks > 1 else 0), 
			filter_channels, n_heads, n_layers, kernel_size, p_dropout,
			window_size=window_size
		)

		self.proj_m = layers.Conv1D(
			# n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1
			n_feats, 1
		)
		self.proj_w = DurationPredictor(
			# n_channels + (spk_emb_dim if n_spks > 1 else 0), 
			filter_channels_dp, kernel_size, p_dropout
		)


	def call(self, x, x_lengths, spk=None):
		x = self.emb(x) * math.sqrt(self.n_channels)
		# x = tf.transpose(x, [0, 2, 1])
		# print(f"x embedding transposed: {x}, {x.shape}")
		# x_mask = tf.expand_dims(
		# 	sequence_mask(x_lengths, tf.shape(x)[2]), 1
		# )
		# x_mask = tf.expand_dims(
		# 	sequence_mask(x_lengths, tf.shape(x)[1]), 1
		# )
		# x_mask = sequence_mask(x_lengths, tf.shape(x)[1])
		x_mask = tf.expand_dims(
			sequence_mask(x_lengths, tf.shape(x)[1]), -1
		)
		x_mask = tf.cast(x_mask, x.dtype)

		x = self.prenet(x, x_mask)
		if self.n_spks > 1:
			x = tf.concat(
				[
					x, 
					tf.tile(
						tf.expand_dims(spk, -1), 
						[1, 1, tf.shape(x)[-1]]
					)
				],
				axis=1
			)
		x = self.encoder(x, x_mask)
		mu = self.proj_m(x) * x_mask

		x_dp = tf.identity(x)
		log_w = self.proj_w(x_dp, x_mask)

		return mu, log_w, x_mask
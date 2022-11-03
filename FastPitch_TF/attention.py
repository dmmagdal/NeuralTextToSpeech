# attention.py


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConvNorm(layers.Layer):
	def __init__(self, dim, kernel_size=1, strides=1, padding=None,
			dilation=1, use_bias=True):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert (kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)
		# padding = "valid" if padding == 0 else "same"
		padding = "same" if padding else "causal"

		self.conv = layers.Conv1D(
			dim, kernel_size=kernel_size, strides=strides,
			padding=padding, dilation_rate=dilation, use_bias=use_bias
		)


	def call(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal


'''
# NOTE: This class is not currently implemented at the moment as
# transliterating the original implementation is rather complex and
# time consuming. In addition, the layer this is applied to
# (ConvAttention) uses it only as a place holder for its projection
# sub-layer. The FastPitch model also sets the ConvAttention layer to
# use "3xconv" for its alignment type. The 3xconv alignment uses a
# series of conv layers instead of the invertible 1x1 convLU layer
# below.
class Invertible1x1ConvLUS(layers.Layer):
	def __init__(self, c):
		super(Invertible1x1ConvLUS, self).__init__()
		# Sample a random orthonormal matrix to initialize weights
		w, _ = tf.linalg.qr(tf.random.normal([c, c]))
		# Ensure determinant is 1.0 and NOT -1.0
		if tf.linalg.det(W) < 0:
			W[:, 0] = -1 * W[:, 0]
		lu, p = tf.linalg.lu(W)

		self.add_weight("p", p)
		# Diagonals of lower will always be 1s anyway
		lower = tf.tril(lower, -1)
'''


class ConvAttention(layers.Layer):
	def __init__(self, n_mel_channels=80, n_speaker_dim=128, 
			n_text_channels=512, n_attn_channels=80, temperature=1.0,
			n_mel_convs=2, align_query_enc_type="3xconv", 
			use_query_proj=True):
		super(ConvAttention, self).__init__()
		self.temperature = temperature
		self.att_scaling_factor = np.sqrt(n_attn_channels)
		# self.softmax = tf.nn.softmax() # Original has dim=3
		# self.log_softmax = tf.nn.log_softmax() # Original has dim=3
		# self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
		self.attn_proj = layers.Conv2D(1, kernel_size=1)
		self.align_query_enc_type = align_query_enc_type
		self.use_query_proj = bool(use_query_proj)

		self.key_proj = keras.Sequential([
			ConvNorm(
				n_text_channels * 2, kernel_size=3, use_bias=True,
			),
			layers.ReLU(),
			ConvNorm(
				n_attn_channels, kernel_size=1, use_bias=True
			)
		])

		self.align_query_enc_type = align_query_enc_type

		if align_query_enc_type == "inv_conv":
			# self.query_proj = Invertible1x1ConvLUS(n_mel_channels)
			raise ValueError("Invertible1x1ConvLUS is not currently implemented or supported at this time.")
		elif align_query_enc_type == "3xconv":
			self.query_proj = keras.Sequential([
				ConvNorm(
					n_mel_channels * 2, kernel_size=3, use_bias=True,
				),
				layers.ReLU(),
				ConvNorm(
					n_mel_channels, kernel_size=1, use_bias=True
				),
				layers.ReLU(),
				ConvNorm(
					n_attn_channels, kernel_size=1, use_bias=True
				)
			])
		else:
			raise ValueError("Unknown query encoder type specified")


	def call(self, queries, keys, query_lens, mask=None, key_lens=None,
			keys_encoded=None, attn_prior=None):
		# Attention mechanism for FLowtron Parallel
		# Unlike in Flowtron, we have no restrictions such as causality
		# etc, since we only need this during training.
		# 
		keys_enc = self.key_proj(keys) # B x n_attn_dims x T2

		# Beware can only do this since query_dim = attn_dim - n_mel_channels
		if self.use_query_proj:
			if self.align_query_enc_type == "inv_conv":
				# queries_enc, log_det_W = self.query_proj(queries)
				raise ValueError("Invertible1x1ConvLUS is not currently implemented or supported at this time.")
			elif self.align_query_enc_type == "3xconv":
				queries_enc = self.query_proj(queries)
				log_det_W = 0.0
			else:
				# queries_enc, log_det_W = self.query_proj(queries) # This still uses inv1x1convlu if I'm not mistaken
				raise ValueError("Invertible1x1ConvLUS is not currently implemented or supported at this time.")
		else:
			queries_enc, log_det_W = queries, 0.0

		# Different ways of computing attention, one is isotopic
		# gaussian (per phoneme) Simplistic Gaussian Isotopic
		# Attention.

		# B x n_attn_dims x T1 x T2
		attn = (queries_enc[:, : , :, None] - keys_enc[:, :, None]) ** 2
		# Compute log likelihood from a gaussian
		attn = -0.0005 * tf.math.reduce_sum(attn, axis=1, keepdims=True)
		if attn_prior is not None:
			attn = tf.nn.log_softmax(attn, axis=3) +\
				tf.math.log(attn_prior[:, None] + 1e-8)

		attn_logprob = tf.identity(attn)

		if mask is not None:
			mask = tf.expand_dims(\
				tf.transpose(mask, perm=[0, 2, 1]), axis=2
			)
			attn = tf.where(mask, -float("inf"), attn)

		attn = tf.nn.softmax(attn, axis=3) # Softmax along T2
		return attn, attn_logprob
# tts.py


import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import monotonic_align
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss
from model.utils import fix_len_compatibility


class GradTTS(keras.Model):
	def __init__(self, n_vocab, n_spkr, spk_emb_dim, n_enc_channels,
			filter_channels, filter_channels_dp, n_heads, n_enc_layers,
			enc_kernel, enc_dropout, window_size, n_mel_channels,
			dec_dim, beta_min, beta_max, pe_scale):
		super(GradTTS, self).__init__()
		self.n_vocab = n_vocab
		self.n_spkr = n_spkr
		self.spk_emb_dim = spk_emb_dim
		self.n_enc_channels = n_enc_channels
		self.filter_channels = filter_channels
		self.filter_channels_dp = filter_channels_dp
		self.n_heads = n_heads
		self.n_enc_layers = n_enc_layers
		self.window_size = window_size
		self.n_mel_channels = n_mel_channels
		self.dec_dim = dec_dim
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.pe_scale = pe_scale

		if n_spkr > 1:
			self.spk_emb = layers.Embedding(n_spkr, spk_emb_dim)
		self.encoder = TextEncoder(
			n_vocab, n_mel_channels, n_enc_channels, filter_channels,
			filter_channels_dp, n_heads, n_enc_layers, enc_kernel,
			enc_dropout, window_size
		)
		self.decoder = Diffusion(
			n_mel_channels, dec_dim, n_spkr, spk_emb_dim, beta_min, 
			beta_max, pe_scale
		)


	def call(self, x, x_lengths, n_timesteps, temperature=1.0, 
			stoc=False, spk=None, length_scale=1.0):
		# Generates mel-spectrogram from text. Returns:
		#	1. encoder outputs
		#	2. decoder outputs
		#	3. generated alignment
		# @param:
		# @param:
		# @param:
		# @param:
		# @param:
		# @param:
		# @param:
		if self.n_spkr > 1:
			# Get speaker embedding.
			spk = self.spk_emb(spk)

		# Get encoder outputs 'mu_x' and log-scalted token duration
		# 'logw'.
		mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

		w = tf.math.exp(logw) * x_mask
		w_ceil = tf.math.ceil(w) * length_scale
		y_lengths = tf.cast(
			tf.clip_by_value(tf.math.reduce_sum(w_ceil, [1, 2]), 1), 
			dtype=tf.int64
		)
		y_max_length = int(tf.math.reduce_max(y_lengths))
		y_max_length_ = fix_len_compatibility(y_max_length)

		# Using obtained 'w' construct alignment map 'attn'.
		y_mask = tf.cast(
			tf.expand_dims(
				sequence_mask(y_lengths, y_max_length_), 1
			),
			dtype=x_mask.dtype
		)
		attn_mask = tf.expand_dims(x_mask, -1) *\
			tf.expand_dims(y_mask, 2)
		attn = tf.expand_dims(
			generate_path(
				tf.squeeze(w_ceil, 1), tf.squeeze(attn_mask, 1)
			),
			1
		)

		# Align encoded text and get mu_y.
		mu_y = tf.linalg.matmul(
			tf.transpose(tf.squeeze(attn, 1), [0, 2, 1]),
			tf.transpose(mu_x, [0, 2, 1])
		)
		mu_y = tf.transpose(mu_y, [0, 2, 1])
		encoder_outputs = mu_y[:, :, :y_max_length]

		# Sample latent representation from terminal distribution
		# N(mu_y, I).
		z = mu_y + tf.random.uniform(mu_y) / temperature
		# Generate sample by performing reverse dynamics.
		decoder_outputs = self.decoder(
			z, y_mask, mu_y, n_timesteps, stoc, spk
		)
		decoder_outputs = decoder_outputs[:, :, :y_max_length]

		return (
			encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]
		)


	def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, 
			out_size=None):
		pass
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
			dec_dim, beta_min, beta_max, pe_scale, out_size):
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
		self.out_size = out_size

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


	# def call(self, x, x_lengths, n_timesteps, temperature=1.0, 
	# 		stoc=False, spk=None, length_scale=1.0):
	def call(self, inputs, temperature=1.0, stoc=False, spk=None, 
			length_scale=1.0):
		# Generates mel-spectrogram from text. Returns:
		#	1. encoder outputs
		#	2. decoder outputs
		#	3. generated alignment
		# @param: x (tf.Tensor), batch of texts, converted to a tensor
		#	with phoneme embedding ids.
		# @param: x_lengths (tf.Tensor), lengths of texts in batch.
		# @param: n_timesteps (int), number of steps to use for reverse
		#	diffusion in decoder.
		# @param: temperature (float, optional), controls variance of
		#	terminal distribution.
		# @param: stoc (bool, optional), flag that adds stochastic term
		#	to the decoder sampler. Usually, does not provide synthesis
		#	improvements.
		# @param: length_scale (float, optional), controls speech pace.
		#	Increase value to slow down generated speech and vice 
		#	versa.
		(x, x_lengths, n_timesteps) = inputs

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


	def train_step(self, batch):
		(
			text_padded, input_lengths, mel_padded, output_lengths,
			speaker_id, 
		) = batch

		print("In train step")
		print(f"batch size: {tf.shape(text_padded)[0]}")

		# tf.print(tf.shape(x[0]))
		# y = (mel_padded, input_lengths, output_lengths)
		# print(f"x: {x}\ny: {y}\nnum_frames: {len_x}")

		with tf.GradientTape() as tape:
			loss = self.compute_loss(
				text_padded, input_lengths, mel_padded, output_lengths,
				out_size=self.out_size
			)
			# y_pred = self(text_padded, input_lengths, 0)
			# print(f"y_pred: {y_pred}")
			# loss, meta = self.loss(y_pred, y)
			exit()
		return {}


	def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, 
			out_size=None):
		# Computes 3 losses:
		#	1. duration loss: loss between predicted token durations 
		#		and those extracted by Monotinic Alignment Search 
		#		(MAS).
		#	2. prior loss: loss between mel-spectrogram and encoder 
		#		outputs.
		#	3. diffusion loss: loss between gaussian noise and its 
		#		reconstruction by diffusion-based decoder.
		# @param: x (tf.Tensor), batch of texts, converted to a tensor 
		#	with phoneme embedding ids.
		# @param: x_lengths (torch.Tensor), lengths of texts in batch.
		# @param: y (tf.Tensor), batch of corresponding 
		#	mel-spectrograms.
		# @param: y_lengths (tf.Tensor), lengths of mel-spectrograms in
		#	batch.
		# @param: out_size (int, optional), length (in mel's sampling 
		#	rate) of segment to cut, on which decoder will be trained.
		#	Should be divisible by 2^{num of UNet downsamplings}. 
		#	Needed to increase batch size.
		print(f"n_spkr: {self.n_spkr}, {spk}")

		if self.n_spkr > 1:
			# Get speaker embedding.
			spk = self.spk_emb(spk)

		# Get encoder outputs 'mu_x' and log-scaled token duration
		# 'logw'.
		mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
			
		y_mask = tf.expand_dims(sequence_mask(y_lengths))
		attn_mask = tf.expand_dims(x_mask, 1) * tf.expand_dims(y_mask, 2)

		# Use MAS to find most likely alignment 'attn' between text and
		# mel-spectrogram.
		const = -0.5 * math.log(2 * math.pi) * self.n_feats
		factor = -0.5 * tf.ones(mu_x.shape, dtype=mu_x.dtype)
		y_square = tf.linalg.matmul(
			tf.transpose(factor, [0, 2, 1]), y ** 2
		)
		y_mu_double = tf.linalg.matmul(
			2.0 * tf.transpose((factor * mu_x), [0, 2, 1]), y
		)
		mu_square = tf.expand_dims(
			tf.math.reduce_sum(factor * (mu_x ** 2), 1), -1
		)
		log_prior = y_square - y_mu_double + mu_square + const

		attn = monotonic_align.maximum_path(
			log_prior, tf.squeeze(attn_mask, -1)
		)

		# Compute loss between predicted log-scaled durations and those
		# obtained from MAS.
		logw_ = tf.math.log(
			1e-8 + tf.math.reduce_sum(tf.expand_dims(attn, -1), 1)
		) * x_mask
		dur_loss = duration_loss(logw, logw_, x_lengths)

		# Cut a small segment of mel-spectrogram in order to increase
		# batch size.
		if not isinstance(out_size, type(None)):
			max_offset = tf.clip_by_value(
				(y_lengths - out_size), 0, tf.float32.max
			)
			offset_ranges = list(zip(
				[0] * max_offset.shape[0], max_offset.numpy()
			))
			out_offset = tf.convert_to_tensor(
				tf.convert_to_tensor(
					random.choice(range(start, end)) 
					if end > start else 0
				),
				dtype=tf.int64
			)

			attn_cut = tf.zeros(
				[attn.shape[0], attn.shape[1], out_size], 
				dtype=attn.dtype
			)
			y_cut = tf.zeros(
				[y.shape[0], self.n_feats, out_size], dtype=y.dtype
			)
			y_cut_lengths = []
			for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
				y_cut_length = out_size +\
					tf.clip_by_value(
						(y_lengths[i] - out_size), tf.float32.min, 0
					)
				y_cut_lengths.append(y_cut_length)
				cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
				y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
				attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
			y_cut_lengths = tf.convert_to_tensor(
				y_cut_lengths, dtype=tf.int64
			)
			y_cut_mask = tf.expand_dims(
				sequence_mask(y_cut_lengths), -1
			)

			attn = attn_cut
			y = y_cut
			y_mask = y_cut_mask

		# Align encoded text with mel-spectrogram and get mu_y segment.
		mu_y = tf.linalg.matmul(
			tf.transpose(tf.squeeze(attn, -1), [0, 2, 1]),
			tf.transpose(mu_x, [0, 2, 1])
		)
		mu_y = tf.transpose(mu_y, [0, 2, 1])

		# Compute loss of score-based decoder.
		diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk)

		# Compute loss between aligned encoder outputs and
		# mel-spectrogram.
		prior_loss = tf.math.reduce_sum(
			0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
		)
		prior_loss = prior_loss / (tf.math.reduce_sum(y_mask) * self.n_feats)

		return dur_loss, prior_loss, diff_loss
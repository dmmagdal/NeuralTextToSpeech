# model.py


from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from common import filter_warnings
from common.layers import ConvReLUNorm
from common.utils import mask_from_lens
# from fastpitch.alignment import b_mas, mas_width1
# from fastpitch.attention import ConvAttention
# from fastpitch.transformer import FFTransformer
from alignment import b_mas, mas_width1
from attention import ConvAttention
from transformer import FFTransformer


def regulate_len(durations, enc_out, pace: float = 1.0,
		mel_max_len: Optional[int] = None):
	# If target=None, then predicted durations are applied
	dtype = enc_out.dtype
	reps = tf.cast(durations, dtype=tf.float32) / pace
	reps = tf.cast(reps + 0.5, dtype=tf.int64)
	dec_lens = tf.math.reduce_sum(reps, axis=1)

	max_len = tf.math.reduce_max(dec_lens)
	pad = [[0, 0], [1, 0]]
	reps_cumsum = tf.math.cumsum(
		tf.pad(reps, pad, constant_values=0.0), axis=1
	)[:, None, :]
	reps_cumsum = tf.cast(reps_cumsum, dtype=dtype)

	range_ = tf.range(max_len)[None, :, None]
	mult = (
		(reps_cumsum[:, :, :-1] <= range_) &\
		(reps_cumsum[:, :, 1:] > range_) 
	)
	mult = tf.cast(mult, dtype=dtype)
	enc_rep = tf.linalg.matmul(mult, enc_out)

	if mel_max_len is not None:
		enc_rep = enc_rep[:, :mel_max_len]
		dec_lens = tf.clip_by_value(
			dec_lens, clip_value_min=tf.float32.min,
			clip_value_max=mel_max_len
		)
	return enc_rep, dec_lens


def average_pitch(pitch, durs):
	durs_cums_ends = tf.cast(
		tf.math.cumsum(durs, axis=1), dtype=tf.int64
	)
	durs_cums_starts = tf.pad(durs_cums_ends[:, :-1], [[1, 0]])
	pitch_nonzero_cums = tf.pad(
		tf.math.cumsum(pitch != 0, axis=2), [[1, 0]]
	)
	pitch_cums = tf.pad(tf.math.cumsum(pitch, axis=2), [[1, 0]])

	batch_size, length = durs_cums_ends.shape
	n_formants = pitch.shape[1]
	dcs = tf.broadcast_to(
		durs_cums_starts[:, None, :], (batch_size, n_formants, 1)
	)
	dce = tf.broadcast_to(
		durs_cums_ends[:, None, :], (batch_size, n_formants, 1)
	)


class TemporalPredictor(layers.Layer):
	def __init__(self, input_size, filter_size, kernel_size, dropout,
			n_layers=2, n_predictions=1):
		super(TemporalPredictor, self).__init__()

		self.layers = keras.Sequential([
			ConvReLUNorm(
				# input_size is i == 0 else filter_size,
				filter_size, kernel_size=kernel_size, dropout=dropout
			)
			for i in range(n_layers)
		])
		self.n_predictions = n_predictions
		self.fc = layers.Dense(self.n_predictions, use_bias=True)


	def call(self, enc_out, enc_out_mask):
		out = enc_out * enc_out_mask
		out = self.layers(out)
		out = self.fc(out) * enc_out_mask
		return out


class FastPitch(keras.Model):
	def __init__(self, n_mel_channels, n_symbols, padding_idx,
			symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
			in_fft_d_head, in_fft_conv1d_kernel_size, 
			in_fft_conv1d_filter_size, in_fft_output_size,
			p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
			out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
			out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
			out_fft_output_size, p_out_fft_dropout, p_out_fft_dropatt,
			p_out_fft_dropemb, dur_predictor_kernel_size,
			dur_predictor_filter_size, p_dur_predictor_dropout,
			dur_predictor_n_layers, pitch_embedding_kernel_size,
			energy_conditioning, energy_predictor_kernel_size,
			energy_predictor_filter_size,
			p_energy_predictor_dropout, energy_predictor_n_layers,
			energy_embedding_kernel_size, n_speakers, 
			speaker_emb_weight, pitch_conditioning_formants=1):
		super(FastPitch, self).__init__()

		self.encoder = FFTransformer(
			n_layer=in_fft_n_layers, n_head=in_fft_d_heads,
			d_model=symbols_embedding_dim,
			d_head=in_fft_d_head, d_inner=in_fft_conv1d_filter_size,
			kernel_size=in_fft_conv1d_kernel_size,
			dropout=p_in_fft_dropout, dropatt=p_in_fft_dropatt,
			dropemb=p_in_fft_dropemb, embed_input=True,
			d_embed=symbols_embedding_dim, n_embed=n_symbols,
			padding_idx=padding_idx
		)

		if n_speakers > 1:
			self.speaker_emb = layers.Embedding(
				n_speakers, symbols_embedding_dim
			)
		else:
			self.speaker_emb = None
		self.speaker_emb_weight = speaker_emb_weight

		self.duration_predictor = TemporalPredictor(
			in_fft_output_size, 
			filter_size=dur_predictor_filter_size,
			kernel_size=dur_predictor_kernel_size,
			dropout=p_dur_predictor_dropout,
			n_layers=dur_predictor_n_layers
		)

		self.decoder = FFTransformer(

		)

		pitch_predictor = TemporalPredictor(
			in_fft_output_size, 
			filter_size=dur_predictor_filter_size,
			kernel_size=dur_predictor_kernel_size,
			dropout=p_dur_predictor_dropout,
			n_layers=dur_predictor_n_layers
		)

		self.pitch_emb = layers.Conv1D(
			symbols_embedding_dim, 
			kernel_size=pitch_embedding_kernel_size,
			padding="same" if int((pitch_embedding_kernel_size - 1) / 2) else "causal"
		)

		# Store values precomputed for training data within the model.
		self.pitch_mean = tf.zeros((1))
		self.pitch_std = tf.zeros((1))

		self.energy_conditioning = energy_conditioning
		if energy_conditioning:
			self.energy_predictor = TemporalPredictor(
				in_fft_output_size, 
				filter_size=energy_predictor_filter_size,
				kernel_size=energy_predictor_kernel_size,
				dropout=p_energy_predictor_dropout,
				n_layers=energy_predictor_n_layers,
				n_predictions=1
			)

			self.energy_emb = layers.Conv1D(
				symbols_embedding_dim,
				kernel_size=energy_embedding_kernel_size,
				padding="same" if int((energy_embedding_kernel_size - 1) / 2) else "causal"
			)

		self.proj = layers.Dense(n_mel_channels, use_bias=True)
		self.attention = ConvAttention(
			n_mel_channels, 0, symbols_embedding_dim,
			use_query_proj=True, align_query_enc_type="3xconv"
		)


	def binarize_attention(self, attn, in_lens, out_lens):
		# For training purposes only. Binarizes attention with MAS.
		# These will no longer recieve a gradient.
		# @param: attn, (batch_size, 1, max_mel_len, max_text_len)
		batch_size = attn.shape[0]
		attn_out_cpu = np.zeros(attn.shape, dtype=np.float32)
		log_attn_cpu = tf.math.log(attn)
		log_attn_cpu = log_attn_cpu.numpy()
		out_lens_cpu = out_lens
		in_lens_cpu = in_lens
		for ind in range(batch_size):
			hard_attn = mas_width1(
				log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]]
			)
			attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
		attn_out = tf.convert_to_tensor(
			attn_out_cpu, dtype=attn.dtype
		)
		return attn_out


	def binarize_attention_parallel(self, attn, in_lens, out_lens):
		# For training purposes only. Binarizes attention with MAS.
		# These will no longer recieve a gradient.
		# @param: attn, (batch_size, 1, max_mel_len, max_text_len)
		log_attn_cpu = tf.math.log(attn).numpy()
		attn_out = b_mas(
			log_attn_cpu, in_lens.numpy(), out_lens.numpy(), width=1
		)
		return tf.convert_to_tensor(attn_out)


	def call(self, inputs, use_gt_pitch=True, pace=1.0, 
			max_duration=75):
		(inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense,
		 speaker, attn_prior, audiopaths) = inputs

		text_max_len = inputs.size(1)
		mel_max_len = mel_tgt.size(2)

		# Calculate speaker embedding
		if self.speaker_emb is None:
			spk_emb = 0
		else:
			spk_emb = tf.expand_dims(self.speaker_emb(speaker), 1)
			# spk_emb.mul_(self.speaker_emb_weight)
			spk_emb = tf.linalg.matmul(
				self.speaker_emb_weight, spk_emb
			)

		# Input FFT
		enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

		# Predict durations
		# log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
		# dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)
		log_dur_pred = tf.squeeze(
			self.duration_predictor(enc_out, enc_mask), -1
		)
		dur_pred = tf.clip_by_value(
			tf.math.exp(log_dur_pred) - 1, 0, max_duration
		)

		# Predict pitch
		pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

		# Alignment
		text_emb = self.encoder.word_emb(inputs)

		# make sure to do the alignments before folding
		attn_mask = mask_from_lens(input_lens, max_len=text_max_len)
		attn_mask = attn_mask[..., None] == 0
		# attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

		attn_soft, attn_logprob = self.attention(
			mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
			key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)

		attn_hard = self.binarize_attention(attn_soft, input_lens, mel_lens)

		# Viterbi --> durations
		attn_hard_dur = attn_hard.sum(2)[:, 0, :]
		dur_tgt = attn_hard_dur
		assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens))

		# Average pitch over characters
		pitch_tgt = average_pitch(pitch_dense, dur_tgt)

		if use_gt_pitch and pitch_tgt is not None:
			pitch_emb = self.pitch_emb(pitch_tgt)
		else:
			pitch_emb = self.pitch_emb(pitch_pred)
		enc_out = enc_out + pitch_emb.transpose(1, 2)

		# Predict energy
		if self.energy_conditioning:
			energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)

			# Average energy over characters
			energy_tgt = average_pitch(energy_dense.unsqueeze(1), dur_tgt)
			energy_tgt = torch.log(1.0 + energy_tgt)

			energy_emb = self.energy_emb(energy_tgt)
			energy_tgt = energy_tgt.squeeze(1)
			enc_out = enc_out + energy_emb.transpose(1, 2)
		else:
			energy_pred = None
			energy_tgt = None

		len_regulated, dec_lens = regulate_len(
			dur_tgt, enc_out, pace, mel_max_len)

		# Output FFT
		dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
		mel_out = self.proj(dec_out)
		return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
				pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard,
				attn_hard_dur, attn_logprob)


	def inference(self, inputs, pace=1.0, dur_tgt=None, pitch_tgt=None,
			energy_tgt=None, pitch_transform=None, max_duration=75,
			speaker=0):
		pass


	def train_step(self, batch):
		(
			text_padded, input_lengths, mel_padded, output_lengths,
			len_x, pitch_padded, energy_padded, speaker_id, 
			attn_prior_padded, audiopath
		) = batch

		# Compute len_x from batch (see collate_fn() from
		# data_function.py for more of an explanation).
		len_x = tf.math.reduce_sum(len_x)
		pass
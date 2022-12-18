# loss_function.py


import tensorflow as tf
from tensorflow import keras
from common.utils import mask_from_lens
# from fastpitch.attn_loss_function import AttentionCTCLoss
from attn_loss_function import AttentionCTCLoss


class FastpitchLoss:
	def __init__(self, dur_predictor_loss_scale=1.0,
			pitch_predictor_loss_scale=1.0, attn_loss_scale=1.0,
			energy_predictor_loss_scale=1.0):
		self.dur_predictor_loss_scale = dur_predictor_loss_scale
		self.pitch_predictor_loss_scale = pitch_predictor_loss_scale
		self.energy_predictor_loss_scale = energy_predictor_loss_scale
		self.attn_loss_scale = attn_loss_scale
		self.attn_ctc_loss = AttentionCTCLoss()


	def call(self, model_out, targets, is_training=True, 
			meta_agg="mean"):
		(
			mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred, 
			pitch_target, energy_pred, energy_target, attn_soft, 
			attn_hard, attn_dur, attn_logprob
		) = model_out

		(mel_target, input_lens, output_lens) = targets

		dur_target = attn_dur
		dur_lens = input_lens

		# (batch_size, Hz, timesteps) -> (batch_size, timesteps, Hz)
		mel_target = tf.transpose([1, 2])

		# Duration loss
		dur_mask = mask_from_lens(
			dur_lens, max_len=dur_target.shape[1]
		)
		log_dur_target = tf.math.log(
			tf.cast(dur_target, dtype=tf.float32) + 1
		)
		loss_fn = keras.losses.MeanSquaredError()
		dur_pred_loss = loss_fn(log_dur_pred, log_dur_target)
		dur_pred_loss = tf.math.reduce_sum(dur_pred_loss * dur_mask) /\
			tf.math.reduce_sum(dur_mask)

		# Mel loss
		ldiff = mel_target.shape[1] - mel_out.shape[1]
		mel_pad = [[0, 0], [0, 1], [0, 0]]
		mel_out = tf.pad(mel_out, mel_pad, constant_values=0.0)
		mel_mask = tf.cast(
			tf.math.not_equal(mel_out, 0), dtype=tf.float32
		)
		loss_fn = keras.losses.MeanSquaredError()
		mel_loss = loss_fn(mel_out, mel_target)
		mel_loss = tf.math.reduce_sum(mel_loss * mel_mask) /\
			tf.math.reduce_sum(mel_mask)

		# Pitch 
		ldiff = pitch_target.shape[2] - pitch_pred.shape[2]
		pitch_pad = [[0, 0], [0, 0], [0, 1]]
		pitch_pred = tf.pad(pitch_pred, pitch_pad, constant_values=0.0)
		pitch_loss = keras.losses.MeanSquaredError()(
			pitch_target, pitch_pred
		)
		pitch_loss = tf.math.reduce_sum(
			pitch_loss * tf.expand_dims(dur_mask, axis=1)
		) / tf.math.reduce_sum(dur_mask)

		# Energy loss 
		if energy_pred is not None:
			energy_pad = [[0, 0], [0, 1]]
			energy_pred = tf.pad(
				energy_pred, energy_pad, constant_values=0.0
			)
			energy_loss = keras.losses.MeanSquaredError()(
				energy_target, energy_pred
			)
			energy_loss = tf.math.reduce_sum(energy_loss * dur_mask) /\
				tf.math.reduce_sum(dur_mask)
		else:
			energy_loss = 0

		# Attention loss
		attn_loss = self.attn_ctc_loss(
			attn_logprob, input_lens, output_lens
		)

		# Total loss
		loss = (
			mel_loss + (dur_pred_loss * self.dur_predictor_loss_scale)
			+ (pitch_loss * self.pitch_predictor_loss_scale)
			+ (energy_loss * self.energy_predictor_loss_scale)
			+ (attn_loss * self.attn_loss_scale)
		)

		meta = {
			"loss": loss,
			"mel_loss": mel_loss,
			"duration_predictor_loss": dur_pred_loss,
			"pitch_loss": pitch_loss,
			"attn_loss": attn_loss,
			"dur_error": tf.math.reduce_sum(
				tf.math.abs(dur_pred - dur_target)
			) / tf.math.reduce_sum(dur_mask),
		}

		if energy_pred is not None:
			meta["energy_loss"] = energy_loss

		assert meta_agg in ('sum', 'mean')
		if meta_agg == 'sum':
			bsz = mel_out.shape[0] # batch_size
			meta = {k: v * bsz for k, v in meta.items()}
		return loss, meta
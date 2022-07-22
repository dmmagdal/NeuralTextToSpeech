# model.py


from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from common import monkey_patch
from common.layers import ConvReLUNorm
from common.utils import mask_from_lens
from fastpitch.alignment import b_mas, mas_width1
from fastpitch.attention import ConvAttention
from fastpitch.transformer import FFTransformer


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
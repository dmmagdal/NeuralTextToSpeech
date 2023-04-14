""" from https://github.com/jaywalnut310/glow-tts """
# utils.py


import tensorflow as tf


def sequence_mask(length, max_length=None):
	if max_length is None:
		max_length = tf.math.reduce_max(length)
	x = tf.range(int(max_length), dtype=length.dtype)
	return tf.expand_dims(x, 0) < tf.expand_dims(length, 1)


def fix_len_compatibility(length, num_downsampling_in_unet=2):
	while True:
		if length % (2 ** num_downsampling_in_unet) == 0:
			return length
		length += 1


def convert_to_shape(pad_shape):
	l = pad_shape[::-1]
	pad_shape = [item for sublist in l for item in sublist]
	return pad_shape


def generate_path(duration, mask):
	b, t_x, t_y = mask.shape
	cum_duration = tf.math.cumsum(duration, 1)
	path = tf.zeros((b, t_x, t_y), dtype=mask.dtype)

	cum_duration_flat = tf.reshape(cum_duration, [b * t_x])
	path = tf.cast(
		sequence_mask(cum_duration_flat, t_y), dtype=mask.dtype
	)
	path = tf.reshape(path, [b, t_x, t_y])
	path = path - tf.pad(
		# path, convert_pad_shape([[0, 0], [1, 0], [0, 0]])[:, :-1] # Original
		path, [[0, 0], [1, 0], [0, 0]]
	)[:, :-1]
	path = path * mask
	return path


def duration_loss(logw, logw_, lengths):
	loss = tf.math.reduce_sum((logw - logw_) ** 2) /\
		tf.math.reduce_sum(tf.cast(lengths, dtype=tf.float32))
	return loss
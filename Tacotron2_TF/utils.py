# utils.py


import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
	max_len = tf.math.reduce_max(lengths)
	ids = tf.cast(tf.range(0, max_len), dtype=tf.float32)
	mask = (ids < tf.expand_dims(lengths, axis=1))
	return mask


def load_wav_to_tf(full_path):
	sampling_rate, data = read(full_path)
	return tf.convert_to_tensor(
		data.astype(np.float32), dtype=tf.float32
	), sampling_rate


def load_filepaths_and_text(filename, split="|"):
	with open(filename, 'r') as f:
		filepaths_and_text = [line.strip().split(split) for line in f]
	return filepaths_and_text
# utils.py


import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read


def get_mask_from_lengths(lengths):
	max_len = tf.math.reduce_max(lengths)
	# ids = tf.cast(tf.range(0, max_len), dtype=tf.float32)
	# mask = (ids < tf.expand_dims(lengths, axis=1))
	mask = tf.sequence_mask(lengths, max_len)
	return mask


def load_wav_to_tensorflow(full_path):
	""" Loads wavdata into tensorflow array """
	file = tf.io.read_file(full_path)

	audio, sampling_rate = tf.audio.decode_wav(file)
	audio = tf.squeeze(audio, axis=-1)

	audio = tf.cast(audio, tf.float32)
	return audio, sampling_rate


def load_filepaths_and_text(filelist, split="|"):
	if isinstance(filelist, str):
		with open(filelist, encoding='utf-8') as f:
			filepaths_and_text = [
				line.strip().split(split) for line in f
			]
	else:
		filepaths_and_text = filelist
	return filepaths_and_text
# tools.py


import numpy as np
import tensorflow as tf


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
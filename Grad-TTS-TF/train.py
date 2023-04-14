# train.py
#
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import params
from data import Data
from model import GradTTS
from utils import plot_tensor, save_plot
from text.symbols import symbols


def main():
	# Set seed.
	tf.random.set_seed(params.seed)
	np.random.seed(params.seed)

	# Initialize logger.

	# Initialize data loaders/dataset.
	dataset_path = './ljspeech_train'
	train_data = Data(
		dataset_path, # dataset_path
		params.train_filelist_path, # filelist_path
		params.cmudict_path, # cmudict_path
		n_mel_channels=params.n_feats,
		n_speakers=params.n_spks,
		load_mel_from_disk=True,#False,
		sampling_rate=params.sample_rate,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		mel_fmin=params.f_min,
		mel_fmax=params.f_max
	)
	train_dataset = tf.data.Dataset.from_generator(
		train_data.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
			tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			tf.TensorSpec(
				# shape=(None, n_mel_channels), dtype=tf.float32
				shape=(None, params.n_feats), dtype=tf.float32
			),														# mel
			tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
		)
	)
	# train_dataset = train_dataset.batch(16, drop_remainder=True)
	train_dataset = train_dataset.batch(8, drop_remainder=True)
	valid_data = Data(
		dataset_path, # dataset_path
		params.valid_filelist_path, # filelist_path
		params.cmudict_path, # cmudict_path
		n_mel_channels=params.n_feats,
		n_speakers=params.n_spks,
		load_mel_from_disk=True,#False,
		sampling_rate=params.sample_rate,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		mel_fmin=params.f_min,
		mel_fmax=params.f_max
	)
	valid_dataset = tf.data.Dataset.from_generator(
		valid_data.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
			tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			tf.TensorSpec(
				# shape=(None, n_mel_channels), dtype=tf.float32
				shape=(None, params.n_feats), dtype=tf.float32
			),														# mel
			tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
		)
	)
	# valid_dataset = train_dataset.batch(16, drop_remainder=True)
	valid_dataset = train_dataset.batch(8, drop_remainder=True)
	test_data = Data(
		dataset_path, # dataset_path
		params.test_filelist_path, # filelist_path
		params.cmudict_path, # cmudict_path
		n_mel_channels=params.n_feats,
		n_speakers=params.n_spks,
		load_mel_from_disk=True,#False,
		sampling_rate=params.sample_rate,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		mel_fmin=params.f_min,
		mel_fmax=params.f_max
	)
	test_dataset = tf.data.Dataset.from_generator(
		test_data.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
			tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			tf.TensorSpec(
				# shape=(None, n_mel_channels), dtype=tf.float32
				shape=(None, params.n_feats), dtype=tf.float32
			),														# mel
			tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
		)
	)

	# Initialize model.
	model = GradTTS(
		params.n_symbols, 1, None, params.n_enc_channels, 
		params.filter_channels, params.filter_channels_dp, 
		params.n_heads, params.n_enc_layers, params.enc_kernel,
		params.enc_dropout, params.window_size, params.n_feats, 
		params.dec_dim, params.beta_min, params.beta_max, 
		params.pe_scale, params.out_size
	)

	# Get number of encoder parameters.

	# Get number of decoder parameters.

	# Get total number of parameters.

	# Initialize optimizer.
	optimizer = Adam(learning_rate=params.learning_rate)

	# Start training.
	model.compile(
		optimizer=optimizer, #loss=[loss, attention_kl_loss],
		run_eagerly=True
	)

	# Build model to get the summary (not a hard requirement but nice
	# to have).
	# batch = list(train_dataset.batch(1).take(1).as_numpy_iterator())
	# print(batch[0])
	# print(batch[0][0])
	# inputs = {"x": tf.squeeze(batch[0][0], 0), "x_lengths": tf.squeeze(batch[0][1], 0), "n_timesteps": 1}
	# model.call(inputs)
	# model.build()
	# model.summary()

	# model.fit(train_dataset, epochs=1, batch_size=16)
	model.fit(train_dataset, epochs=1)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
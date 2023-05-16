# train.py


from argparse import ArgumentParser
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data import Data
from model import DiffWave
from params import params


def main():
	# Parse arguments.
	parser = ArgumentParser(description='train (or resume training) a DiffWave model')
	# parser.add_argument('model_dir',
	# 	help='directory in which to store model checkpoints and training logs')
	# parser.add_argument('data_dirs', nargs='+',
	# 	help='space separated list of directories from which to read .wav files for training')	# Not used. Hard coded.
	parser.add_argument('--max_steps', default=None, type=int,
		help='maximum number of training steps')
	parser.add_argument('--fp16', action='store_true', default=False,
		help='use 16-bit floating point operations for training')
	args = parser.parse_args()


	# Initialize datasets.
	gtzan = False

	dataset_path = './ljspeech_train_valid'
	train_list = './filelists/ljs_audio_text_train_v3.txt'
	valid_list = './filelists/ljs_audio_text_val.txt'

	# Output signature for datasets.
	signature = (
		tf.TensorSpec(shape=(None,), dtype=tf.float32),		# audio
		tf.TensorSpec(
			shape=(None, params.n_mels), dtype=tf.float32
		),													# mel
	)
	if params.unconditional:
		# Output signature is different if using unconditional model.
		signature = (
			tf.TensorSpec(shape=(None,), dtype=tf.float32),	# audio
			tf.TensorSpec(shape=(), dtype=tf.float32)		# mel
		)
	if gtzan:
		# Output signature is different if using gtzan.
		signature = (
			tf.TensorSpec(shape=(1, None,), dtype=tf.float32),	# audio
			tf.TensorSpec(shape=(), dtype=tf.float32)			# mel
		)

	train_data = Data(
		dataset_path, # dataset_path
		train_list, # filelist_path
		params,
		# n_mel_channels=n_mel_channels,
		n_mel_channels=params.n_mels,
		# n_speakers=params.n_spks,
		n_speakers=1, # Manually added because this tests LJSpeech dataset
		load_mel_from_disk=False,
		# sampling_rate=sampling_rate,
		sampling_rate=params.sample_rate,
		# filter_length=filter_length,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		# mel_fmin=mel_fmin,
		# mel_fmax=mel_fmax
		mel_fmin=params.f_min,
		mel_fmax=params.f_max, 
		from_gtzan=gtzan
	)

	valid_data = Data(
		dataset_path, # dataset_path
		valid_list, # filelist_path
		params,
		# n_mel_channels=n_mel_channels,
		n_mel_channels=params.n_mels,
		# n_speakers=params.n_spks,
		n_speakers=1, # Manually added because this tests LJSpeech dataset
		load_mel_from_disk=False,
		# sampling_rate=sampling_rate,
		sampling_rate=params.sample_rate,
		# filter_length=filter_length,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		# mel_fmin=mel_fmin,
		# mel_fmax=mel_fmax
		mel_fmin=params.f_min,
		mel_fmax=params.f_max, 
		from_gtzan=gtzan
	)

	train_dataset = tf.data.Dataset.from_generator( # Use in eager execution.
		train_data.generator,
		args=(),
		output_signature=signature
	)

	valid_dataset = tf.data.Dataset.from_generator( # Use in eager execution.
		valid_data.generator,
		args=(),
		output_signature=signature
	)

	train_dataset = train_dataset.batch(params.batch_size, drop_remainder=True)
	train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
	valid_dataset = valid_dataset.batch(params.batch_size, drop_remainder=True)
	valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)

	# Initialize optimizer and loss.
	optimizer = keras.optimizers.Adam(
		learning_rate=params.learning_rate
	)
	loss = keras.losses.MeanAbsoluteError()

	# Initialize model.
	model = DiffWave(params)

	# Compute the number of epochs from max_steps (1 step = 1 batch).
	steps_per_epoch = train_data.__len__() / params.batch_size
	epochs = math.ceil(args.max_steps / steps_per_epoch)
	print(f"Training DiffWave vocoder for {args.max_steps} steps at batch size {params.batch_size} ({epochs} epochs)")

	# Compile and train model.
	model.compile(optimizer=optimizer, loss=loss, )#run_eagerly=True)
	history = model.fit(
		train_dataset, 
		# validation_data=valid_dataset, 
		epochs=epochs
	)
	model.summary()

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	# with tf.device('/cpu:0'):
	# 	main()
	main()
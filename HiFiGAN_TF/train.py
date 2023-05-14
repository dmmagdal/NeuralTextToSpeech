# train.py


import math
import os
import json
import argparse
import tensorflow as tf
from tensorflow import keras
from data import Data
from gan import HiFiGAN
from hparams import HParams
from model import Generator
from model import MultiPeriodDiscriminator, MultiScaleDiscriminator
from model import get_generator, get_mpd, get_msd


# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)


def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--group_name', default=None)                                     # Not needed/used 
	parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
	parser.add_argument('--input_mels_dir', default='ft_dataset')
	# parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')     # Not needed/used (replaced)
	# parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt') # Not needed/used (replaced)
	parser.add_argument('--checkpoint_path', default='cp_hifigan')
	parser.add_argument('--config', default='')
	parser.add_argument('--training_epochs', default=3100, type=int)
	# parser.add_argument('--stdout_interval', default=5, type=int)                         # Not needed/used 
	# parser.add_argument('--checkpoint_interval', default=5000, type=int)                  # Not needed/used 
	# parser.add_argument('--summary_interval', default=100, type=int)                      # Not needed/used 
	# parser.add_argument('--validation_interval', default=1000, type=int)                  # Not needed/used 
	parser.add_argument('--fine_tuning', default=False, type=bool)

	# Hard coded stuff (for testing).

	# Load the hyperparameters from the config file.
	config_file = "./config_v1.json"
	config_file = "./config_v2.json"
	config_file = "./config_v3.json"
	with open(config_file, "r") as f:
		hparams = json.load(f)

	hparams = HParams(**hparams)

	frames_per_seg = math.ceil(
		hparams.segment_size / hparams.hop_size
	)

	# Load the training and validation datasets.
	train_filelist = './filelists/ljs_audio_text_train_v3.txt'
	train_dataset = Data(
		train_filelist,
		segment_size=hparams.segment_size,
		n_mel_channels=hparams.num_mels,
		filter_length=hparams.n_fft,
		hop_length=hparams.hop_size,
		win_length=hparams.win_size,
		sampling_rate=hparams.sampling_rate,
		mel_fmin=hparams.fmin,
		mel_fmax=hparams.fmax,
		n_speakers=1, # Manually set because config.json does not have variable (& it isnt necessary for the vocoder)
	)
	train_data = tf.data.Dataset.from_generator( # Use in eager execution.
		train_dataset.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel
			tf.TensorSpec(shape=(None,), dtype=tf.float32),						# audio
			tf.TensorSpec(shape=(), dtype=tf.string),							# filename
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel_loss
		)
		# output_signature=(
		# 	tf.TensorSpec(shape=(frames_per_seg, hparams.num_mels), dtype=tf.float32),	# mel
		# 	tf.TensorSpec(shape=(hparams.segment_size,), dtype=tf.float32),				# audio
		# 	tf.TensorSpec(shape=(), dtype=tf.string),									# filename
		# 	tf.TensorSpec(shape=(frames_per_seg, hparams.num_mels), dtype=tf.float32),	# mel_loss
		# )
	)

	valid_filelist = './filelists/ljs_audio_text_val.txt'
	valid_dataset = Data(
		valid_filelist,
		segment_size=hparams.segment_size,
		n_mel_channels=hparams.num_mels,
		filter_length=hparams.n_fft,
		hop_length=hparams.hop_size,
		win_length=hparams.win_size,
		sampling_rate=hparams.sampling_rate,
		mel_fmin=hparams.fmin,
		mel_fmax=hparams.fmax,
		split=False,
		shuffle=False,
		n_speakers=1, # Manually set because config.json does not have variable (& it isnt necessary for the vocoder)
	)
	valid_data = tf.data.Dataset.from_generator( # Use in eager execution.
		valid_dataset.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel
			tf.TensorSpec(shape=(None,), dtype=tf.float32),						# audio
			tf.TensorSpec(shape=(), dtype=tf.string),							# filename
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel_loss
		)
		# output_signature=(
		# 	tf.TensorSpec(shape=(frames_per_seg, hparams.num_mels), dtype=tf.float32),	# mel
		# 	tf.TensorSpec(shape=(hparams.segment_size,), dtype=tf.float32),						# audio
		# 	tf.TensorSpec(shape=(), dtype=tf.string),							# filename
		# 	tf.TensorSpec(shape=(frames_per_seg, hparams.num_mels), dtype=tf.float32),	# mel_loss
		# )
	)

	# For mixed precision training.
	# train_data = train_data.map(
	# 	lambda x, y, f, z: (tf.cast(x, policy.compute_dtype), tf.cast(y, policy.compute_dtype), f, tf.cast(z, policy.compute_dtype))
	# )
	# valid_data = valid_data.map(
	# 	lambda x, y, f, z: (tf.cast(x, policy.compute_dtype), tf.cast(y, policy.compute_dtype), f, tf.cast(z, policy.compute_dtype))
	# )

	train_data = train_data.batch(
		hparams.batch_size, drop_remainder=True
	)
	train_data = train_data.prefetch(tf.data.AUTOTUNE)
	valid_data = valid_data.batch(1, drop_remainder=True) # Original train.py specifies batch_size of 1.
	valid_data = valid_data.prefetch(tf.data.AUTOTUNE)

	# Uncomment for eager execution debugging.
	# print(list(train_data.as_numpy_iterator())[0])
	# print(list(valid_data.as_numpy_iterator())[0])

	# Initialize the models.
	generator = Generator(hparams)
	mpd = MultiPeriodDiscriminator()
	msd = MultiScaleDiscriminator()

	# gen = get_generator((None, hparams.num_mels), hparams)
	# mpd2 = get_mpd((None,), (None,))
	# msd2 = get_msd((None,), (None,))
	# gen.summary()
	# mpd2.summary()
	# msd2.summary()

	gan = HiFiGAN(hparams, generator, mpd, msd)

	# Initialize optimizers.
	optim_g = keras.optimizers.Adam(
		learning_rate=hparams.learning_rate,
		beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
	)
	optim_mpd = keras.optimizers.Adam(
		learning_rate=hparams.learning_rate,
		beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
	)
	optim_msd = keras.optimizers.Adam(
		learning_rate=hparams.learning_rate,
		beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
	)

	# Compile the model.
	gan.compile(optim_g, optim_mpd, optim_msd, run_eagerly=True)

	# Train the model.
	gan.fit(train_data, validation_data=valid_data, epochs=1)
	
	# Exit the program.
	exit()


if __name__ == '__main__':
	# with tf.device('/cpu:0'):
	# 	main()
	main()

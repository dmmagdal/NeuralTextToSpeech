# train.py


from argparse import ArgumentParser
from datetime import datetime
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
	parser.add_argument('--resume_training', action='store_true', default=False,
		help='resume from latest checkpoint')
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
	if args.max_steps is not None:
		steps_per_epoch = train_data.__len__() / params.batch_size
		epochs = math.ceil(args.max_steps / steps_per_epoch)
		print(f"Training DiffWave vocoder for {args.max_steps} steps at batch size {params.batch_size} ({epochs} epochs)")
	else:
		epochs = 1
		print(f"Training DiffWave vocoder for 1 epoch at batch size {params.batch_size}")

	# Tensorboard callback.

	tensorboard_dir = "./diffwave_logs/fit/"
	tensorboard_log = os.path.join(
		tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
	)
	tensorboard_callback = keras.callbacks.TensorBoard(
		log_dir=tensorboard_log, histogram_freq=1
	) # https://www.tensorflow.org/tensorboard/get_started

	# Checkpoint callback.
	checkpoint_dir = "./diffwave_logs/checkpoints/"
	checkpoint_log = os.path.join(
		checkpoint_dir, "diff_wave-{epoch:04d}"
	) # saved model
	# os.makedirs(checkpoint_log, exist_ok=True)
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		checkpoint_log,
		# monitor="val_loss", # Default. Can change to regular "loss"
		# save_best_only=True,
		save_freq="epoch",
	) # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

	# Logic for resuming training from latest checkpoint or train from
	# scratch.
	if args.resume_training and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
		# Load model weights with tf.train.latest_checkpoint.
		# latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		# initial_epoch = int(
		# 	latest_checkpoint.lstrip(checkpoint_dir + "diff_wave-")
		# )
		latest_checkpoint = os.path.join(
			checkpoint_dir,
			os.listdir(checkpoint_dir)[-1] # assume lexicographic order with 4 digit epoch value
		)
		initial_epoch = int(
			latest_checkpoint.lstrip(checkpoint_dir + "diff_wave-")
		)
	else:
		# No previous checkpoint exits. Create checkpoint directory and
		# start at the first epoch.
		os.makedirs(checkpoint_dir, exist_ok=True)
		latest_checkpoint = None
		initial_epoch = 0

	if latest_checkpoint:
		# Load latest checkpoint to model (should have loss and
		# optimizer state).
		print(f"loading from checkpoint {latest_checkpoint}")
		model = tf.keras.models.load_model(latest_checkpoint)
		# print(model.summary())
		# exit()
	else:
		# Compile model.
		model.compile(optimizer=optimizer, loss=loss,)# run_eagerly=True)

	epochs = 5#3 # Hard coded. Set to 5 after run.
	
	# Train model.
	history = model.fit(
		train_dataset, 
		validation_data=valid_dataset, 
		epochs=epochs,
		callbacks=[tensorboard_callback, checkpoint_callback],
		initial_epoch=initial_epoch
	)
	model.summary()

	# Save and load the model (use SavedModel format).
	model.save('diff_wave') # default format is SavedModel for Tf 2.X
	loaded_model = keras.models.load_model('diff_wave')

	# Use loaded model summary to confirm model matches.
	loaded_model.summary()

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	# with tf.device('/cpu:0'):
	# 	main()
	main()
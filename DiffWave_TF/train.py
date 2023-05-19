# train.py


from argparse import ArgumentParser
from datetime import datetime
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
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
	parser.add_argument('--save_weights', action='store_true', default=False,
		help='use and save weights of model when training')
	args = parser.parse_args()

	###################################################################
	# Initialize dataset.
	###################################################################

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

	###################################################################
	# Model, optimizer, & loss initialization.
	###################################################################

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


	###################################################################
	# Initialize callbacks.
	###################################################################

	# Callbacks list and callbacks folder.
	callback_logdir = "./diffwave_logs"
	callback_list = []

	# Tensorboard callback.
	tensorboard_dir = os.path.join(callback_logdir, "fit/")
	tensorboard_log = os.path.join(
		tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
	)
	tensorboard_callback = keras.callbacks.TensorBoard(
		log_dir=tensorboard_log, histogram_freq=1
	)
	callback_list.append(tensorboard_callback)

	# Checkpoint callback.
	checkpoint_dir = os.path.join(callback_logdir, "checkpoints/")
	checkpoint_save = "diff_wave-{epoch:04d}"
	if args.save_weights:
		checkpoint_save += ".h5"
	checkpoint_log = os.path.join(
		checkpoint_dir, checkpoint_save
	)
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		checkpoint_log,
		# monitor="val_loss", # Default. Can change to regular "loss"
		# save_best_only=True,
		save_weights_only=args.save_weights,
		save_freq="epoch",
	) 
	callback_list.append(checkpoint_callback)

	###################################################################
	# Load model from checkpoints and set starting epoch.
	###################################################################

	# Logic for resuming training from latest checkpoint or train from
	# scratch.
	if args.resume_training and os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
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

	# Load model from latest checkpoint or compile the model if
	# training from scratch.
	if latest_checkpoint:
		# Load latest checkpoint to model (should have loss and
		# optimizer state).
		print(f"loading from checkpoint {latest_checkpoint}")
		model = tf.keras.models.load_model(latest_checkpoint)

		if not args.save_weights:
			# Load optimizer if model is a savedModel.
			optimizer = model.optimizer
	else:
		# There is no latest checkpoint. The model is being trained
		# from scratch. Compile model.
		model.compile(optimizer=optimizer, loss=loss,)# run_eagerly=True)

	epochs = 1#3 # Hard coded. Set to 5 after run.

	###################################################################
	# Train the model.
	###################################################################

	if args.save_weights:
		# Train model.
		history = model.fit(
			train_dataset, 
			validation_data=valid_dataset, 
			epochs=epochs,
			# callbacks=[tensorboard_callback, checkpoint_callback],
			callbacks=callback_list,
			initial_epoch=initial_epoch
		)
		model.summary()
	else:
		# Compile callback list if using custom train loop/SavedModel
		# format.
		callbacks = keras.callbacks.CallbackList(
			callback_list, add_history=True, model=model
		)
		logs = {}
		callbacks.on_train_begin(logs=logs)

		# Keep for plotting.
		train_loss_results = []
		valid_loss_results = []
		val_loss = keras.losses.MeanAbsoluteError()

		for epoch in range(initial_epoch, epochs):
			# Step through training data.
			for train in train_dataset:
				step(train, model, optimizer, loss, params, True)

			# Step through validation data.
			for valid in valid_dataset:
				step(valid, model, optimizer, val_loss, params)

			# End of epoch.
			train_loss_results.append(loss.result())
			valid_loss_results.append(val_loss.result())
			print(f"Epoch {epoch + 1}/{epochs} loss: {loss.result()}, val_loss: {val_loss.result()}")

		# Save and load the model (use SavedModel format).
		model.save('diff_wave') # default format is SavedModel for Tf 2.X
		loaded_model = keras.models.load_model('diff_wave')

		# Use loaded model summary to confirm model matches.
		loaded_model.summary()

		items = list(valid_dataset.take(1).as_numpy_iterator())[0]
		audio, mel = items
		N, T = audio.shape
		t = tf.random.uniform([N], 0, len(params.noise_schedule))
		t = tf.cast(tf.round(t), dtype=tf.int32)
		prediction = loaded_model((audio, t, mel))
		prediction = loaded_model.predict((audio, t, mel))
		prediction = loaded_model.call((audio, t, mel))
		# loaded_model.train_step(items) # gave error due to input shape
		prediction = loaded_model.predict_on_batch((audio, t, mel))

	# Exit the program.
	exit(0)


def step(data, model, optimizer, loss_fn, params, training=None):
	# Unpack data.
	audio, mel = data
	N, T = audio.shape

	# Noise.
	beta = np.array(params.noise_schedule)
	noise_level = np.cumprod(1 - beta)
	noise_level = tf.convert_to_tensor(noise_level)
	t = tf.random.uniform([N], 0, len(params.noise_schedule))
	t = tf.cast(tf.round(t), dtype=tf.int32)
	# noise_scale = tf.expand_dims(noise_level[t], 1)
	noise_scale = tf.expand_dims(tf.gather(noise_level, t), 1)
	noise_scale = tf.cast(noise_scale, dtype=tf.float32) # added to convert from float64 to float32
	noise_scale_sqrt = noise_scale ** 0.5
	noise = tf.random.normal(tf.shape(audio), dtype=tf.float32)
	noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
	
	if training:
		with tf.GradientTape() as tape:
			# predicted = self(noisy_audio, t, mel)
			# predicted = self((noisy_audio, t), spectrogram=mel)
			predicted = model.call((noisy_audio, t, mel), training=training)
			# loss = loss(noise, tf.squeeze(predicted, 1)) # Original
			loss = loss_fn(noise, tf.squeeze(predicted, -1))
	else:
		predicted = model.call((noisy_audio, t, mel), training=training)
		loss = loss_fn(noise, tf.squeeze(predicted, -1))

	# Compute gradients.
	# trainable_vars = model.trainable_variables
	gradients = tape.gradient(loss, model.trainable_vars)

	# Update weights.
	optimizer.apply_gradients(zip(gradients, model.trainable_vars))

	# Update metrics (loss).
	loss_fn.update_state(
		# noise, tf.squeeze(predicted, 1) # Original
		noise, tf.squeeze(predicted, -1)
	)


if __name__ == "__main__":
	# with tf.device('/cpu:0'):
	# 	main()
	main()
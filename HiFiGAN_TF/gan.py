# gan.py


import os
import tensorflow as tf
from tensorflow import keras
from common.audio_processing_tf import STFT
from losses import discriminator_loss, feature_loss, generator_loss


class HiFiGAN(keras.Model):
	def __init__(self, hparams, generator, mpd, msd):
		super(HiFiGAN, self).__init__()
		# Hyperparameters
		self.hparams = hparams

		# Models.
		self.generator = generator
		self.mpd = mpd
		self.msd = msd

		# Optimizers
		self.optim_g = keras.optimizers.Adam(
			learning_rate=hparams.learning_rate,
			beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
		)
		self.optim_mpd = keras.optimizers.Adam(
			learning_rate=hparams.learning_rate,
			beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
		)
		self.optim_mpd = keras.optimizers.Adam(
			learning_rate=hparams.learning_rate,
			beta_1=hparams.adam_b1, beta_2=hparams.adam_b2
		)

		# Losses.
		self.disc_loss = discriminator_loss
		self.gen_loss = generator_loss
		self.feature_loss = feature_loss
		self.l1_loss = keras.losses.MeanAbsoluteError()

		# STFT.
		self.stft = STFT(
			filter_length=hparams.n_fft, 
			frame_step=hparams.hop_size, 
			frame_length=hparams.win_size, 
			sampling_rate=hparams.sampling_rate,
			mel_fmin=hparams.fmin, mel_fmax=hparams.fmax
		)


	def build(self):
		self.generator.build(input_shape=(None, None, self.hparams.num_mels)) # (batch_size, len, n_mels)
		self.mpd.build(input_shape=(None, None,))
		# self.mpd.build(input_shape=((None, None,), (None, None,))) # ()
		# self.mpd.build(input_shape1=(None, None,), input_shape2=(None, None,)) # ()
		# self.msd.build(input_shape=((None, None,), (None, None,)))


	def call(self, audio):
		return tf.cast(self.generator(audio), dtype=tf.int16)


	def train_step(self, batch):
		x, y, _, y_mel = batch

		y = tf.expand_dims(y, -1)

		with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
			# Generator.
			y_g_hat = self.generator(x, training=True)
			y_g_hat_mel = self.stft.mel_spectrogram(y_g_hat)

			# MPD.
			y_df_hat_r, y_df_hat_g, _, _ = self.mpd(
				y, tf.identity(y_g_hat)
			)
			loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(
				y_df_hat_r, y_df_hat_g
			)

			# MSD.
			y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(
				y, tf.identity(y_g_hat)
			)
			loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(
				y_ds_hat_r, y_ds_hat_g
			)

			# Discriminator loss.
			loss_disc_all = loss_disc_s + loss_disc_f

			# Generator loss.
			loss_mel = self.l1_loss(y_mel, y_g_hat_mel) * 45

			y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(
				y, y_g_hat, training=True
			)
			y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(
				y, y_g_hat, training=True
			)
			loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
			loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
			loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
			loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)
			loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

			# Mel-spectrogram loss.
			mel_error = self.l1_loss(y_mel, y_g_hat_mel)

		# Compute Gradients.
		gen_gradients = g_tape.gradient(
			loss_gen_all, self.generator.trainable_variables
		)
		mpd_gradients = d_tape.gradient(
			loss_disc_all, self.mpd.trainable_variables
		)
		msd_gradients = d_tape.gradient(
			loss_disc_all, self.msd.trainable_variables
		)

		# Apply Gradients.
		self.generator.optimizer.apply_gradients(
			zip(gen_gradients, self.generator.trainable_variables)
		)
		self.mpd.optimizer.apply_gradients(
			zip(mpd_gradients, self.mpd.trainable_variables)
		)
		self.msd.optimizer.apply_gradients(
			zip(msd_gradients, self.msd.trainable_variables)
		)

		pass

	
	def test_step(self, batch):
		x, y, _, y_mel = batch
		# val_error = 0

		# Compute predictions.
		y_g_hat = self.generator(x)
		y_g_hat_mel = self.stft.mel_spectrogram(y_g_hat)

		# Updates the metrics tracking the loss.
		# self.compiled_loss(y, y_pred, regularization_losses=self.losses)
		val_error = self.l1_loss(y_mel,  y_g_hat_mel)

		# Update the metrics.
		# self.compiled_metrics.update_state(y, y_pred)

		# Return a dict mapping metric names to current value.
		# Note that it will include the loss (tracked in self.metrics).
		# return {m.name: m.result() for m in self.metrics}


	def compile(self):
		self.generator.compile(optimizer=self.optim_g)
		self.mpd.compile(optimizer=self.optim_d)
		self.msd.compile(optimizer=self.optim_d)


	def summary(self):
		print("Generator")
		self.generator.summary()
		print("Multi Period Discriminator")
		self.mpd.summary()
		print("Multi Scale Discriminator")
		self.msd.summary()


	def save(self, path_dir):
		os.makedirs(path_dir, exist_ok=True)
		self.generator.save(os.path.join(path_dir, 'hifiGAN_generator'))
		self.mpd.save(os.path.join(path_dir, 'hifiGAN_mpd_discriminator'))
		self.msd.save(os.path.join(path_dir, 'hifiGAN_msd_discriminator'))


	def load(self, path_dir):
		if not os.path.exists(path_dir):
			FileNotFoundError(f'Parent path {path_dir} does not exist')
		if not os.path.isdir(path_dir):
			FileExistsError(f'Path {path_dir} already exists as a file & is not a valid directory')
		pass
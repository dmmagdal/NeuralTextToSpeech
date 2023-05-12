# gan.py


import os
import tensorflow as tf
from tensorflow import keras
from common.audio_processing_tf import STFT
from losses import discriminator_loss, feature_loss, generator_loss


class HiFiGAN(keras.Model):
	def __init__(self, hparams, generator, mpd, msd, **kwargs):
		super(HiFiGAN, self).__init__(**kwargs)
		# Hparams.
		self.hparams = hparams

		# Models.
		self.generator = generator
		self.mpd = mpd
		self.msd = msd

		# STFT.
		self.stft = STFT(
			filter_length=hparams.n_fft, 
			frame_step=hparams.hop_size, 
			frame_length=hparams.win_size, 
			sampling_rate=hparams.sampling_rate,
			mel_fmin=hparams.fmin, mel_fmax=hparams.fmax
		)


	def train_step(self, batch):
		x, y, _, y_mel = batch

		y = tf.expand_dims(y, -1)

		with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
			# Generator.
			y_g_hat = self.generator(x, training=True)
			y_g_hat_mel = self.stft.mel_spectrogram(y_g_hat)

			# MPD.
			y_df_hat_r, y_df_hat_g, _, _ = self.mpd(
				(y, tf.identity(y_g_hat)), training=True
			)
			loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.discriminator_loss(
				y_df_hat_r, y_df_hat_g
			)

			# MSD.
			y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(
				(y, tf.identity(y_g_hat)), training=True
			)
			loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.discriminator_loss(
				y_ds_hat_r, y_ds_hat_g
			)

			# Discriminator loss.
			loss_disc_all = loss_disc_s + loss_disc_f

			# Generator loss.
			loss_mel = self.l1_loss(y_mel, y_g_hat_mel) * 45

			y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(
				(y, y_g_hat), training=True
			)
			y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(
				(y, y_g_hat), training=True
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
		# self.generator.optimizer.apply_gradients( # Test later
		# 	zip(gen_gradients, self.generator.trainable_variables)
		# )
		# self.mpd.optimizer.apply_gradients(
		# 	zip(mpd_gradients, self.mpd.trainable_variables)
		# )
		# self.msd.optimizer.apply_gradients(
		# 	zip(msd_gradients, self.msd.trainable_variables)
		# )
		self.optim_g.apply_gradients( # Similar to MelGAN example
			zip(gen_gradients, self.generator.trainable_variables)
		)
		self.optim_mpd.apply_gradients(
			zip(mpd_gradients, self.mpd.trainable_variables)
		)
		self.optim_msd.apply_gradients(
			zip(msd_gradients, self.msd.trainable_variables)
		)

		# Update metrics/trackers.
		self.gen_loss_tracker.update_state(loss_gen_all)
		self.disc_loss_tracker.update_state(loss_disc_all)
		self.mel_loss_tracker.update_state(mel_error)

		return {
			"gen_loss": self.gen_loss_tracker.result(),
			"disc_loss": self.disc_loss_tracker.result(), 
			"mel_Loss": self.mel_loss_tracker.result(),
		}

	
	def test_step(self, batch):
		x, y, _, y_mel = batch
		# val_error = 0

		# Dynamic pad.
		# max_mel_len = tf.shape(x)
		# print(max_mel_len)
		# print(tf.shape(y))
		# print(tf.shape(y_mel))
		# exit()

		# Compute predictions.
		y_g_hat = self.generator(x)
		y_g_hat_mel = self.stft.mel_spectrogram(y_g_hat)

		# Updates the metrics tracking the loss.
		# self.compiled_loss(y, y_pred, regularization_losses=self.losses)
		val_error = self.l1_loss(y_mel,  y_g_hat_mel)

		# Update the metrics.
		# self.compiled_metrics.update_state(y, y_pred)
		self.val_loss_tracker.update(val_error)

		# Return a dict mapping metric names to current value.
		# Note that it will include the loss (tracked in self.metrics).
		# return {m.name: m.result() for m in self.metrics}
		return {
			"val_mel_loss": self.val_loss_tracker.result(),
		}



	def compile(self, optim_g, optim_mpd, optim_msd, **kwargs):
		super().compile(**kwargs)

		# Losses.
		self.disc_loss = discriminator_loss
		self.gen_loss = generator_loss
		self.feature_loss = feature_loss
		self.l1_loss = keras.losses.MeanAbsoluteError()

		# Optimizers.
		# self.generator.compile(optimizer=optim_g) # Test later
		# self.mpd.compile(optimizer=optim_mpd)
		# self.msd.compile(optimizer=optim_msd)
		self.optim_g = optim_g # Similar to MelGAN example
		self.optim_mpd = optim_mpd
		self.optim_msd = optim_msd

		# Metrics/trackers.
		self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
		self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
		self.mel_loss_tracker = keras.metrics.Mean(name="mel_loss")
		self.val_loss_tracker = keras.metrics.Mean(name="val_mel_loss")


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
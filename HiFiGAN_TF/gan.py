# gan.py


import os
import tensorflow as tf
from tensorflow import keras


class HiFiGAN(keras.Model):
	def __init__(self, hparams, generator, mpd, msd, optim_g, optim_d):
		super(HiFiGAN, self).__init__()
		self.hparams = hparams
		self.generator = generator
		self.mpd = mpd
		self.msd = msd
		self.optim_g = optim_g
		self.optim_d = optim_d


	def train_step(self, batch):
		x, y, _, y_mel = batch

		y = tf.expand_dims(y, -1)

		with tf.GradientTape() as g_tape:
			y_g_hat = self.generator(x, training=True)
			# y_g_hat_mel = mel_spectrogram(
			#     tf.squeeze(y_g_hat, -1), self.hparams.n_fft, 
			#     self.hparams.num_mels, self.hparams.sampling_rate, 
			#     self.hparams.hop_size, self.hparams.win_size,
			#     self.hparams.fmin, self.hparams.fmax_for_loss
			# )

		with tf.GradientTape() as mpd_tape:
			y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat)
			# loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
			# 	y_df_hat_r, y_df_hat_g
			# )

		with tf.GradientTape() as msd_tape:
			y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat)
			# loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
			# 	y_ds_hat_r, y_ds_hat_g
			# )

		# loss_disc_all = loss_disc_s + loss_disc_f
		pass


	def compile(self):
		self.generator.compile(optimizer=self.optim_g)
		self.mpd.compile(optimizer=self.optim_d)
		self.msd.compile(optimizer=self.optim_d)
		pass


	def summary(self):
		print("Generator")
		self.generator.summary()
		print("Multi PeriodDiscriminator")
		self.mpd.summary()
		print("Multi Scale Discriminator")
		self.msd.summary()


	def save(self, path):
		pass


	def load(self, path):
		pass
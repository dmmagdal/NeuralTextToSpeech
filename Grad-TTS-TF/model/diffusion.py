# diffusion.py


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Mish(layers.Layer):
	def __init__(self):
		super(Mish, self).__init__()
		self.softplus = layers.Activation('softplus')


	def call(self, x):
		return x * tf.math.tanh(self.softplus(x))


class UpSample(layers.Layer):
	def __init__(self, dim):
		super(UpSample, self).__init__()
		self.conv = layers.Conv2DTranspose(dim)


	def call(self, x):
		return self.conv(x)


class DownSample(layers.Layer):
	def __init__(self, dim):
		super(DownSample, self).__init__()
		self.conv = layers.Conv2D(dim, 3, 2, "same")


	def call(self, x):
		return self.conv(x)


class ReZero(layers.Layer):
	def __init__(self, fn):
		self.fn = fn
		self.g = self.add_weight("g", 1, "zeros")


	def call(self, x):
		return self.fn(x) * self.g


class Block(layers.Layer):
	def __init__(self):
		pass


class ResnetBlock(layers.Layer):
	def __init__(self):
		pass


class LinearAttention(layers.Layer):
	def __init__(self):
		pass


class Residual(layers.Layer):
	def __init__(self):
		pass


class SinusoidalPosEmb(layers.Layer):
	def __init__(self):
		pass


class GradLogPEstimator2D(layers.Layer):
	def __init__(self):
		pass


def get_noise(t, beta_init, beta_term, cumulative=False):
	if cumulative:
		noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t ** 2)
	else:
		noise = beta_init + (beta_term - beta_init) * t
	return noise


class Diffusion(keras.Model):
	def __init__(self, n_mel_channels, dim, n_spkrs=1, spk_emb_dim=64,
			beta_min=0.05, beta_max=20, pe_scale=1000):
		super(Diffusion, self).__init__()
		self.n_mel_channels = n_mel_channels
		self.dim = dim
		self.n_spkrs = n_spkrs
		self.spk_emb_dim = spk_emb_dim
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.pe_scale = pe_scale

		self.estimator = GradLogPEstimator2D(
			dim, n_spkrs=n_spkrs, spk_emb_dim=spk_emb_dim, 
			pe_scale=pe_scale
		)


	def forward_diffusion(self, x0, mask, mu, t):
		time = tf.expand_dims(tf.expand_dims(t, -1), -1)
		cum_noise = get_noise(
			time, self.beta_min, self.beta_max, cumulative=True
		)
		mean = x0 * tf.math.exp(-0.5 * cum_noise) +\
			mu * (1.0 - tf.math.exp(-0.5 * cum_noise))
		variance = 1.0 - tf.math.exp(-cum_noise)
		z = tf.random.normal(x0.shape, dtype=x0.dtype)
		xt = mean + x * tf.math.sqrt(variance)
		return xt *  mask, z * mask


	def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
		h = 1.0 / n_timesteps
		xt = z * mask
		for i in range(n_timesteps):
			t = (1.0 - (i + 0.5) * h) * tf.ones(
				z.shape[0], dtype=z.dtype
			)
			time = tf.expand_dims(tf.expand_dims(t, -1), -1)
			noise_t = get_noise(
				time, self.beta_min, self.beta_max, cumulative=False
			)
			if stoc: # Adds stochastic term
				dxt_det = 0.5 * (mu - xt) - self.estimator(
					xt, mask, mu, t, spk
				)
				dxt_det = dxt_det * noise_t * h
				dxt_stoc = tf.random.normal(z.shape, dtype=z.dtype)
				dxt_stoc = dxt_stoc * tf.math.sqrt(noise_t * h)
				dxt = dxt_det + det_stoc
			else:
				dxt = 0.5 * (
					mu - xt - self.estimator(xt, mask, mu, t, spk)
				)
				dxt = dxt * noise_t * h
			xt = (xt - dxt) * mask
		return xt


	def call(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
		return self.reverse_diffusion(
			z, mask, mu, n_timesteps, stoc, spk
		)


	def loss_t(self, x0, mask, mu, t, spk=None):
		xt, z = self.forward_diffusion(x0, mask, mu, t)
		time = tf.expand_dims(tf.expand_dims(t, -1), -1)
		cum_noise = get_noise(
			time, self.beta_min, self.beta_max, cumulative=True
		)
		noise_estimation = self.estimator(xt, mask, mu, t, spk)
		noise_estimation *= tf.math.sqrt(1.0 - tf.math.exp(-cum_noise))
		loss = tf.math.reduce_sum((noise_estimation + z) ** 2) /\
			(tf.math.reduce_sum(mask) * self.n_mel_channels)
		return loss, xt


	def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
		t = tf.random.uniform(x0.shape[0], dtype=x0.dtype)
		t = tf.clip_by_value(t, offset, 1.0 - offset)
		return self.loss_t(x0, mask, mu, t, spk)
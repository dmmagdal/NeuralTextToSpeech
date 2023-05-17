# model.py


from math import sqrt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DilatedConv1D(layers.Layer):
	def __init__(self, filters, kernel_size, padding="valid", dilation=1, zero_init=False, **kwargs):
		super(DilatedConv1D, self).__init__(**kwargs)
		if padding == 0:
			padding = "valid"
		elif padding == 1:
			padding = "causal"
		else:
			padding = "same"

		# Supposedly the equivalent of torch.nn.init.kaiming_normal_()
		# in tensorflow as suggested by ChatGPT.
		initializer = keras.initializers.VarianceScaling(
			scale=2.0,
			mode="fan_in",
			distribution="truncated_normal"
		) if not zero_init else keras.initializers.Zeros()

		self.conv = layers.Conv1D(
			filters, kernel_size, padding=padding, 
			dilation_rate=dilation,
			kernel_initializer=initializer
		)

	
	def call(self, x):
		return self.conv(x)
	

def silu(x):
	return x * tf.math.sigmoid(x)
	

class DiffusionEmbedding(layers.Layer):
	def __init__(self, max_steps, **kwargs):
		super(DiffusionEmbedding, self).__init__(**kwargs)
		self.embedding = self._build_embedding(max_steps)
		self.projection1 = layers.Dense(512)
		self.projection2 = layers.Dense(512)


	def call(self, diffusion_step):
		if diffusion_step.dtype in [tf.int32, tf.int64]:
			# x = self.embedding[diffusion_step] # Original
			x = tf.gather(self.embedding, diffusion_step)
		else:
			x = self._lerp_embedding(diffusion_step)
		x = self.projection1(x)
		x = silu(x)
		x = self.projection2(x)
		x = silu(x)
		return x


	def _lerp_embedding(self, t):
		low_idx = tf.math.floor(t)
		high_idx = tf.math.ceil(t)
		low = self.embedding[low_idx]
		high = self.embedding[high_idx]
		return low + (high - low) * (t - low_idx)


	def _build_embedding(self, max_steps):
		steps = tf.expand_dims(tf.range(max_steps, dtype=tf.float32), 1)	# [T,1]
		dims = tf.expand_dims(tf.range(64, dtype=tf.float32), 0)			# [1,64]
		table = steps * 10.0 ** (dims * 4.0 / 63.0)     # [T,64]
		table = tf.concat(
			[tf.math.sin(table), tf.math.cos(table)], axis=1
		)
		return table


class SpectrogramUpsampler(layers.Layer):
	def __init__(self, n_mels, **kwargs):
		super(SpectrogramUpsampler, self).__init__(**kwargs)
		self.conv1 = layers.Conv2DTranspose(
			# 1, kernel_size=[3, 32], strides=[1, 16], padding="same" # Original
			1, kernel_size=[32, 3], strides=[16, 1], padding="same"
		)
		self.conv2 = layers.Conv2DTranspose(
			# 1, kernel_size=[3, 32], strides=[1, 16], padding="same" # Original
			1, kernel_size=[32, 3], strides=[16, 1], padding="same"
		)
		self.leaky_relu1 = layers.LeakyReLU(0.4)
		self.leaky_relu2 = layers.LeakyReLU(0.4)

	
	def call(self, x):
		x = tf.expand_dims(x, -1)
		x = self.conv1(x)
		x = self.leaky_relu1(x)
		x = self.conv2(x)
		x = self.leaky_relu2(x)
		x = tf.squeeze(x, -1)
		return x


class ResidualBlock(layers.Layer):
	def __init__(self, n_mels, residual_channels, dilation, uncond=False, **kwargs):
		super(ResidualBlock, self).__init__(**kwargs)
		self.dilated_conv = DilatedConv1D(
			2 * residual_channels, 3, padding=dilation, 
			dilation=dilation
		)
		self.diffusion_projection = layers.Dense(residual_channels)

		if not uncond:
			# Conditional model.
			self.conditioner_projection = DilatedConv1D(
				2 * residual_channels, 1
			)
		else:
			# Unconditional model.
			self.conditioner_projection = None
		
		self.output_projection = DilatedConv1D(
			2 * residual_channels, 1
		)


	def call(self, x, diffusion_step, conditioner=None):
		assert (conditioner is None and self.conditioner_projection is None) or\
			(conditioner is not None and self.conditioner_projection is not None)
		
		diffusion_step = tf.expand_dims(
			# self.diffusion_projection(diffusion_step), -1 # Original
			self.diffusion_projection(diffusion_step), 1
		)
		y = x + diffusion_step

		if self.conditioner_projection is None:
			# Using a unconditional model.
			y = self.dilated_conv(y) 
		else:
			conditioner = self.conditioner_projection(conditioner)
			y = self.dilated_conv(y) + conditioner

		gate, filter = tf.split(y, 2, axis=-1)
		y = tf.math.sigmoid(gate) * tf.math.tanh(filter)

		y = self.output_projection(y)
		residual, skip = tf.split(y, 2, axis=-1)
		return (x + residual) / sqrt(2.0), skip


class DiffWave(keras.Model):
	def __init__(self, params, **kwargs):
		super(DiffWave, self).__init__(**kwargs)
		self.params = params
		self.input_projection = DilatedConv1D(params.residual_channels, 1)
		self.diffusion_embedding = DiffusionEmbedding(
			len(params.noise_schedule)
		)

		if self.params.unconditional: 
			# Use unconditional model.
			self.spectrogram_upsampler = None
		else:
			self.spectrogram_upsampler = SpectrogramUpsampler(
				params.n_mels
			)

		self.residual_layers = [
			ResidualBlock(
				params.n_mels, params.residual_channels, 
				2 ** (i % params.dilation_cycle_length), 
				uncond=params.unconditional
			)
			for i in range(params.residual_layers)
		]
		self.skip_projection = DilatedConv1D(params.residual_channels, 1)
		self.output_projection = DilatedConv1D(1, 1, zero_init=True)

		self.relu1 = layers.ReLU()
		self.relu2 = layers.ReLU()

		# Was originally in learner.py in original repo (could be in
		# train.py in this repo).
		beta = np.array(self.params.noise_schedule)
		noise_level = np.cumprod(1 - beta)
		self.noise_level = noise_level


	# def call(self, audio, diffusion_step, spectrogram=None):
	# def call(self, inputs, training=None, spectrogram=None):
	def call(self, inputs):
		assert len(inputs) < 4 and len(inputs) > 1
		if len(inputs) == 2:
			audio, diffusion_step = inputs
			spectrogram = None
		else:
			audio, diffusion_step, spectrogram = inputs

		assert (spectrogram is None and self.spectrogram_upsampler is None) or \
			(spectrogram is not None and self.spectrogram_upsampler is not None)
		x = tf.expand_dims(audio, -1)
		x = self.input_projection(x)
		x = self.relu1(x)

		diffusion_step = self.diffusion_embedding(diffusion_step)
		if self.spectrogram_upsampler: 
			# Use conditional model.
			spectrogram = self.spectrogram_upsampler(spectrogram)

		skip = None
		for layer in self.residual_layers:
			x, skip_connection = layer(x, diffusion_step, spectrogram)
			skip = skip_connection if skip is None else skip_connection + skip

		x = skip / sqrt(len(self.residual_layers))
		x = self.skip_projection(x)
		x = self.relu2(x)
		x = self.output_projection(x)
		return x
	

	def train_step(self, data):
		audio, mel = data
		N, T = audio.shape

		t = tf.random.uniform([N], 0, len(self.params.noise_schedule))
		t = tf.cast(tf.round(t), dtype=tf.int32)
		# noise_scale = tf.expand_dims(self.noise_level[t], 1)
		noise_scale = tf.expand_dims(tf.gather(self.noise_level, t), 1)
		noise_scale = tf.cast(noise_scale, dtype=tf.float32) # added to convert from float64 to float32
		noise_scale_sqrt = noise_scale ** 0.5
		noise = tf.random.normal(tf.shape(audio), dtype=tf.float32)
		noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
		
		with tf.GradientTape() as tape:
			# predicted = self(noisy_audio, t, mel)
			# predicted = self((noisy_audio, t), spectrogram=mel)
			predicted = self((noisy_audio, t, mel))
			# loss = self.compiled_loss(noise, tf.squeeze(predicted, 1)) # Original
			loss = self.compiled_loss(noise, tf.squeeze(predicted, -1))

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(
			# noise, tf.squeeze(predicted, 1) # Original
			noise, tf.squeeze(predicted, -1)
		)

		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}


	def test_step(self, data):
		audio, mel = data
		N, T = audio.shape

		t = tf.random.uniform([N], 0, len(self.params.noise_schedule))
		t = tf.cast(tf.round(t), dtype=tf.int32)
		noise_scale = tf.expand_dims(tf.gather(self.noise_level, t), 1)
		noise_scale = tf.cast(noise_scale, dtype=tf.float32) # added to convert from float64 to float32
		noise_scale_sqrt = noise_scale ** 0.5
		noise = tf.random.normal(tf.shape(audio), dtype=tf.float32)
		noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
		
		predicted = self((noisy_audio, t, mel))
		loss = self.compiled_loss(noise, tf.squeeze(predicted, -1))

		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(
			noise, tf.squeeze(predicted, -1)
		)

		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}
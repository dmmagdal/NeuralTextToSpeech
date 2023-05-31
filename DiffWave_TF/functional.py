# functional.py
# Implement the Diffwave model using Tensorflow's Functional API. This
# will require the use of a custom training loop and not a train_step()
# since we are not overrieding/sub-classing tf.keras.Model.


from math import sqrt
from model import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def getResNet(n_mels, residual_channels, dilation, uncond=False):
	# Initialize layers.
	input_x = layers.Input((None, None)) # Need to verify input layers.
	x = input_x
	input_t = layers.Input(())
	diffusion_step = input_t
	input_mel = layers.Input((None, n_mels))
	conditioner = input_mel

	dilated_conv = DilatedConv1D(
		2 * residual_channels, 3, padding=dilation, 
		dilation=dilation
	)
	diffusion_projection = layers.Dense(residual_channels)

	if not uncond:
		# Conditional model.
		conditioner_projection = DilatedConv1D(
			2 * residual_channels, 1
		)
	else:
		# Unconditional model.
		conditioner_projection = None
	
	output_projection = DilatedConv1D(
		2 * residual_channels, 1
	)

	# Data pass through layers.
	diffusion_step = tf.expand_dims(
		# diffusion_projection(diffusion_step), -1 # Original
		diffusion_projection(diffusion_step), 1
	)
	y = x + diffusion_step

	if conditioner_projection is None:
		# Using a unconditional model.
		y = dilated_conv(y) 
	else:
		conditioner = conditioner_projection(conditioner)
		y = dilated_conv(y) + conditioner

	gate, filter = tf.split(y, 2, axis=-1)
	y = tf.math.sigmoid(gate) * tf.math.tanh(filter)

	y = output_projection(y)
	residual, skip = tf.split(y, 2, axis=-1)
	# return (x + residual) / sqrt(2.0), skip
	outputs_array = [(x + residual) / sqrt(2.0), skip]

	# Build the model.
	if not uncond:
		inputs_array = [input_x, input_t, input_mel]
		post_fix = "_conditional"
	else:
		inputs_array = [input_mel, input_t]
		post_fix = "_unconditional"
	return keras.Model(
		inputs=inputs_array, outputs=outputs_array, name=f"resnet{post_fix}"
	)


def getDiffwave(params):
		# Initialize layers.
		input_audio = layers.Input(
			shape=(None,), dtype=tf.float32, name="signal_input"
		)
		input_t = layers.Input(
			shape=(), dtype=tf.int32, name="diffusion_step_input"
		)
		input_mel = layers.Input(
			shape=(None, params.n_mels), dtype=tf.float32, 
			name="mel_input"
		)
		diffusion_step = input_t

		input_projection = DilatedConv1D(params.residual_channels, 1)
		diffusion_embedding = DiffusionEmbedding(
			len(params.noise_schedule)
		)

		if params.unconditional: 
			# Use unconditional model.
			spectrogram_upsampler = None
		else:
			# Use conditional model.
			spectrogram_upsampler = SpectrogramUpsampler(
				params.n_mels
			)

		residual_layers = [
			ResidualBlock(
				params.n_mels, params.residual_channels, 
				2 ** (i % params.dilation_cycle_length), 
				uncond=params.unconditional
			)
			for i in range(params.residual_layers)
		]
		skip_projection = DilatedConv1D(params.residual_channels, 1)
		output_projection = DilatedConv1D(1, 1, zero_init=True)

		relu1 = layers.ReLU()
		relu2 = layers.ReLU()

		# Data pass through layers.
		x = tf.expand_dims(input_audio, -1)
		x = input_projection(x)
		x = relu1(x)

		diffusion_step = diffusion_embedding(diffusion_step)
		if spectrogram_upsampler: 
			# Use conditional model.
			spectrogram = spectrogram_upsampler(input_mel)
		else:
			# Use unconditional model.
			spectrogram = None

		skip = None
		for layer in residual_layers:
			x, skip_connection = layer(x, diffusion_step, spectrogram)
			skip = skip_connection if skip is None else skip_connection + skip

		x = skip / sqrt(len(residual_layers))
		x = skip_projection(x)
		x = relu2(x)
		x = output_projection(x)
		# return x

		# Build the model.
		if spectrogram_upsampler:
			inputs_array = [input_audio, input_t, input_mel]
			post_fix = "_conditional"
		else:
			inputs_array = [input_mel, input_t]
			post_fix = "_unconditional"
		return keras.Model(
			inputs=inputs_array, outputs=[x], name=f"diffwave{post_fix}"
		)
# model.py

from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import get_mask_from_lengths
'''
from stft import STFT
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from librosa.filters import mel as librosa_mel_fn
'''


class LinearNorm(layers.Layer):
	def __init__(self, dims, bias=True, activation=None):
		super(LinearNorm, self).__init__()
		self.linear_layer = layers.Dense(
			dims, use_bias=bias, activation=activation
		)


	def call(self, x):
		return self.linear_layer(x)


class ConvNorm(layers.Layer):
	def __init__(self, channels, kernel_size=1, stride=1, padding=None,
			dilation=1, bias=True, activation=None):
		super(ConvNorm, self).__init__()
		# print(f"padding {padding}")
		if padding is None:
			assert kernel_size % 2 == 1
			padding = int(dilation * (kernel_size - 1) / 2)
		else:
			padding = int(dilation * (padding - 1) / 2)
			# print(f"padding2 {padding}")
		padding = "same" if padding else "causal"#"valid"
		# print(f"padding3 {padding}")

		self.conv = layers.Conv1D(
			channels, kernel_size=kernel_size, strides=stride,
			padding=padding, dilation_rate=dilation, use_bias=bias,
			activation=activation
			# activation="relu"
		)


	def call(self, signal):
		return self.conv(signal)


'''
class TacotronSTFT:
	def __init__(self, filter_length=1024, hop_length=256, 
			win_length=1024, n_mel_channels=80, sampling_rate=22050,
			mel_fmin=0.0, mel_fmax=8000.0):
		self.n_mel_channels = n_mel_channels
		self.sampling_rate = sampling_rate
		self.stft_fn = STFT(filter_length, hop_length, win_length)
		mel_basis = librosa_mel_fn(
			sampling_rate, filter_length, n_mel_channels, mel_fmin,
			mel_fmax
		)
		self.mel_basis = tf.convert_to_tensor(
			mel_basis, dtype=tf.float32
		)


	def spectral_normalize(self, magnitudes):
		output = dynamic_range_compression(magnitudes)
		return output


	def spectral_de_normalize(self, magnitudes):
		output = dynamic_range_decompression(magnitudes)
		return output


	# Compute mel-spectrograms from a batch of wavs
	# @param: y, (tf.tensor) with shape (batch_size, timesteps) in
	#	range [-1, 1].
	# @return: returns mel_output (tf.tensor) with 
	#	(batch_size, n_mel_channels, timesteps).
	def mel_spectrogram(self, y):
		assert tf.math.reduce_min(y, axis=-1) >= -1
		assert tf.math.reduce_max(y, axis=-1) <= 1

		magnitudes, phases = self.stft_fn.transform(y)
		mel_output = tf.linalg.matmul(self.mel_basis, magnitudes)
		mel_output = self.spectral_normalize(mel_output)
		return mel_output
'''


class LocationLayer(layers.Layer):
	def __init__(self, attention_n_filters, attention_kernel_size,
			attention_dim):
		super(LocationLayer, self).__init__()
		padding = int((attention_kernel_size - 1) / 2)
		# padding = "same" if padding else "valid"
		self.location_conv = ConvNorm(
			attention_n_filters, kernel_size=attention_kernel_size,
			padding=padding, bias=False, stride=1, dilation=1
		)
		self.location_dense = LinearNorm(
			attention_dim, bias=False, activation="tanh"
		)


	def call(self, attention_weights_cat):
		processed_attention = self.location_conv(attention_weights_cat)
		processed_attention = self.location_dense(processed_attention)
		return processed_attention


class Attention(layers.Layer):
	def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
			attention_location_n_filters, 
			attention_location_kernel_size):
		super(Attention, self).__init__()
		self.query_layer = LinearNorm(
			attention_dim, bias=False, activation="tanh"
		)
		self.memory_layer = LinearNorm(
			attention_dim, bias=False, activation="tanh"
		)
		self.v = LinearNorm(1, bias=False)
		self.location_layer = LocationLayer(
			attention_location_n_filters, 
			attention_location_kernel_size,
			attention_dim
		)
		self.score_mask_value = -float("inf")


	# @param: query, decoder output (batch_size, n_mel_channels * 
	#	n_frames_per_step).
	# @param: processed_memory, processed encoder outputs (batch_size,
	#	timesteps_in, attention_dim).
	# @param: attention_weights_cat, cumulative and previous attention
	#	weights (batch_size, 2, max_timesteps).
	# @return: alignment (batch_size, max_timesteps).
	def get_alignment_energies(self, query, processed_memory,
			attention_weights_cat):
		processed_query = self.query_layer(tf.expand_dims(query, 1))
		processed_attention_weights = self.location_layer(
			attention_weights_cat
		)
		print(f"processed_query {processed_query}, shape {processed_query.shape}")
		print(f"processed_attention_weights {processed_attention_weights}, shape {processed_attention_weights.shape}")
		energies = self.v(tf.math.tanh(
			processed_query + processed_attention_weights +\
			processed_memory
		))
		energies = tf.squeeze(energies, -1)
		return energies


	# @param: attention_hidden_state, attention rnn last output.
	# @param: memory, encoder outputs.
	# @param: processed_memory, processed encoder inputs.
	# @param: attention_weights_act, previous and cumulative attention
	#	weights.
	# @param: mask, binary mask for padded data.
	def call(self, attention_hidden_state, memory, processed_memory,
			attention_weights_cat, mask):
		print(f"Attention block:")
		print(f"attention_hidden_state {attention_hidden_state}, shape {attention_hidden_state.shape}")
		print(f"memory {memory}, shape {memory.shape}")
		print(f"processed_memory {processed_memory}, shape {processed_memory.shape}")
		print(f"attention_weights_cat {attention_weights_cat}, shape {attention_weights_cat.shape}")
		print(f"mask {mask}, shape {mask}")
		alignment = self.get_alignment_energies(
			attention_hidden_state, processed_memory, 
			attention_weights_cat
		)
		print(f"alignment {alignment}, shape {alignment.shape}")

		if mask is not None:
			alignment = tf.where(mask, alignment, self.score_mask_value)
			print(f"Mask is not None")
			print(f"masked alignment {alignment}, shape {alignment.shape}")

		attention_weights = tf.nn.softmax(alignment, axis=1)
		attention_context = tf.linalg.matmul(
			tf.expand_dims(attention_weights, axis=1), memory
		)
		attention_context = tf.squeeze(attention_context, axis=1)

		return attention_context, attention_weights


class Prenet(layers.Layer):
	def __init__(self, sizes):
		super(Prenet, self).__init__()
		# self.layers = [
		# 	[
		# 		LinearNorm(size, bias=False), layers.ReLU(), 
		# 		layers.Dropout(0.5)
		# 	]
		# 	for size in sizes
		# ]
		prenet_layers = []
		for i in range(len(sizes)):
			prenet_layers += [
				LinearNorm(sizes[i], bias=False), 
				layers.ReLU(),
				layers.Dropout(0.5),
			]
		self.prenet = keras.Sequential(prenet_layers)


	def call(self, x, training=None):
		# for linear in self.layers:
		# 	x = linear[2](linear[1](linear[0](x)))
		# return x
		return self.prenet(x)


class Postnet(layers.Layer):
	def __init__(self, hparams):
		super(Postnet, self).__init__()
		self.convolutions = []

		self.convolutions.append(
			keras.Sequential([
				ConvNorm(
					hparams.postnet_embedding_dim, 
					kernel_size=hparams.postnet_kernel_size,
					stride=1,
					padding=int((hparams.postnet_kernel_size - 1) / 2),
					dilation=1, 
					activation="tanh"
				),
				layers.BatchNormalization()
			])
		)
		self.convolutions.append(
			keras.Sequential([
				layers.Activation("tanh"),
				layers.Dropout(0.5)
			])
		)

		for i in range(1, hparams.postnet_n_convolutions - 1):
			self.convolutions.append(
				keras.Sequential([
					ConvNorm(
						hparams.postnet_embedding_dim, 
						kernel_size=hparams.postnet_kernel_size,
						stride=1,
						padding=int((hparams.postnet_kernel_size - 1) / 2),
						dilation=1,
						activation="tanh"
					),
					layers.BatchNormalization()
				])
			)

		self.convolutions.append(
			keras.Sequential([
				ConvNorm(
					hparams.n_mel_channels, 
					kernel_size=hparams.postnet_kernel_size,
					stride=1,
					padding=int((hparams.postnet_kernel_size - 1) / 2),
					dilation=1
				),
				layers.BatchNormalization()
			])
		)

		self.convolutions.append(layers.Dropout(0.5))
		self.postnet = keras.Sequential(self.convolutions)


	def call(self, x):
		return self.postnet(x)


class Encoder(layers.Layer):
	def __init__(self, hparams):
		super(Encoder, self).__init__()

		convolutions = []
		for _ in range(hparams.encoder_n_convolutions):
			conv_layer = keras.Sequential([
				ConvNorm(
					hparams.encoder_embedding_dim,
					kernel_size=hparams.encoder_kernel_size,
					stride=1,
					padding=int((hparams.encoder_kernel_size - 1) / 2),
					dilation=1,
					activation="relu"
				),
				layers.BatchNormalization(),
				layers.ReLU(),
				layers.Dropout(0.5)
			])
			convolutions.append(conv_layer)
		self.convolutions = keras.Sequential(convolutions)

		self.lstm = layers.Bidirectional(
			layers.LSTM(
				int(hparams.encoder_embedding_dim / 2),
				return_sequences=True, # allow for return of same shape tensor in pytorch implementation
				# return_sequences=True makes each cell per timestep emit a signal.
			)
		)

		self.masking = layers.Masking()
		# TODO: Delete this commented out line
		# self.rnn_lstm = layers.RNN(self.lstm, return_sequences=True)


	def call(self, x, input_lengths, training=None):
		print(f"encoder inputs shape pre ConvBlocks: {x.shape}")
		x = self.convolutions(x)
		print(f"encoder data shape post ConvBlocks: {x.shape}")

		# TODO: Delete this commented out block
		# if training:
		# 	x = self.masking(x)
		# 	# x = self.rnn_lstm(x, mask=tf.sequence_mask(input_lengths))
		# 	outputs = self.lstm(x, mask=tf.sequence_mask(input_lengths))
		# else:
		# 	outputs = self.lstm(x)

		outputs = self.lstm(x)
		return outputs


class Decoder(layers.Layer):
	def __init__(self, hparams):
		super(Decoder, self).__init__()
		self.n_mel_channels = hparams.n_mel_channels
		self.n_frames_per_step = hparams.n_frames_per_step
		self.encoder_embedding_dim = hparams.encoder_embedding_dim
		self.attention_rnn_dim = hparams.attention_rnn_dim
		self.decoder_rnn_dim = hparams.decoder_rnn_dim
		self.prenet_dim = hparams.prenet_dim
		self.max_decoder_steps = hparams.max_decoder_steps
		self.gate_threshold = hparams.gate_threshold
		self.p_attention_dropout = hparams.p_attention_dropout
		self.p_decoder_dropout = hparams.p_decoder_dropout

		self.prenet = Prenet(
			[hparams.prenet_dim, hparams.prenet_dim]
		)

		self.attention_rnn = layers.LSTMCell(
			hparams.attention_rnn_dim#, return_state=True
		)

		self.attention_layer = Attention(
			hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
			hparams.attention_dim, hparams.attention_location_n_filters,
			hparams.attention_location_kernel_size
		)

		self.decoder_rnn = layers.LSTMCell(
			hparams.decoder_rnn_dim#, return_state=True
		)

		self.linear_projection = LinearNorm(
			hparams.n_mel_channels * hparams.n_frames_per_step
		)

		self.gate_layer = LinearNorm(1, bias=True, activation="sigmoid")

		self.attn_dropout = layers.Dropout(self.p_attention_dropout)
		self.decoder_dropout = layers.Dropout(self.p_decoder_dropout)
	

	def get_go_frame(self, memory):
		batch_size = tf.shape(memory)[0]
		decoder_input = tf.zeros(
			[batch_size, self.n_mel_channels * self.n_frames_per_step]
		)
		return decoder_input


	def initialize_decoder_states(self, memory, mask):
		batch_size = tf.shape(memory)[0]
		max_time = tf.shape(memory)[1]

		self.attention_hidden = tf.zeros(
			(batch_size, self.attention_rnn_dim)
		)
		self.attention_cell = tf.zeros(
			(batch_size, self.attention_rnn_dim)
		)

		self.decoder_hidden = tf.zeros(
			(batch_size, self.decoder_rnn_dim)
		)
		self.decoder_cell = tf.zeros(
			(batch_size, self.decoder_rnn_dim)
		)

		self.attention_weights = tf.zeros(
			(batch_size, max_time)
		)
		self.attention_weights_cum = tf.zeros(
			(batch_size, max_time)
		)
		self.attention_context = tf.zeros(
			(batch_size, self.encoder_embedding_dim)
		)

		self.memory = memory
		self.processed_memory = self.attention_layer.memory_layer(
			memory
		)
		self.mask = mask

		print(f"memory shape {memory.shape}")
		print(F"MAX_TIME {max_time}")
		print(f"initial attention_hidden shape: {self.attention_hidden.shape}")
		print(f"initial attention_cell shape: {self.attention_cell.shape}")

		print(f"initial decoder_hidden shape: {self.decoder_hidden.shape}")
		print(f"initial attention_hidden shape: {self.decoder_cell.shape}")

		print(f"initial attention_weights shape: {self.attention_weights.shape}")
		print(f"initial attention_weights_cum shape: {self.attention_weights_cum.shape}")
		print(f"initial attention_context shape: {self.attention_context.shape}")

		print(f"initial memory shape: {self.memory.shape}")
		print(f"initial processed_memory shape: {self.processed_memory.shape}")
		if mask is None:
			print(f"initial mask is none")
		else:
			print(f"initial mask shape: {self.mask.shape}")


	def parse_decoder_inputs(self, decoder_inputs):
		# (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
		# decoder_inputs = tf.transpose(decoder_inputs, [1, 2])
		# decoder_inputs = tf.transpose(decoder_inputs, [0, 2, 1]) # tf.transpose requires specifying all dimensions in permutation
		# decoder_inputs (mel-spec) is already in shape (batch_size,
		# mel_len, n_mel_channels) format.
		decoder_inputs = tf.reshape(
			decoder_inputs, 
			(
				tf.shape(decoder_inputs)[0], 
				int(tf.shape(decoder_inputs)[1] / self.n_frames_per_step),
				-1
			)
		)
		# (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
		# decoder_inputs = tf.transpose(decoder_inputs, [0, 1])
		decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2])
		return decoder_inputs


	def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
		alignments = tf.transpose(tf.stack(alignments), [0, 1])

		gate_outputs = tf.transpose(tf.stack(gate_outputs), [0, 1])

		mel_outputs = tf.transpose(tf.stack(mel_outputs), [0, 1])

		mel_outputs = tf.reshape(
			tf.shape(mel_outputs)[0], -1, self.n_mel_channels
		)
		mel_outputs = tf.transpose(mel_outputs, [1, 2])
		return mel_outputs, gate_outputs, alignments


	def decode(self, decoder_input):
		cell_input = tf.concat((decoder_input, self.attention_context), -1)
		print(f"attention_cell {self.attention_cell}, shape {self.attention_cell.shape}")
		print(f"attention_hidden {self.attention_hidden}, shape {self.attention_hidden.shape}")
		# self.attention_hidden, self.attention_cell = self.attention_rnn(
		# 	cell_input, [self.attention_hidden, self.attention_cell]
		# )
		outputs, (self.attention_hidden, self.attention_cell) = self.attention_rnn(
			cell_input, states=[self.attention_hidden, self.attention_cell]
		) # Not entirely sure if I'm doing this right. Refer to Tensorflow tf.keras.layers.LSTMCell vs Pytorch torch.nn.LSTMCell documentation
		self.attention_hidden = self.attn_dropout(
			self.attention_hidden, training=self.training
		)
		print(self.attention_hidden.shape)
		print(self.attention_cell.shape)

		print(f"attention_weights {self.attention_weights}, shape {self.attention_weights.shape}")
		print(f"attention_weights_cum {self.attention_weights_cum}, shape {self.attention_weights_cum.shape}")
		attention_weights_cat = tf.concat(
			(
				tf.expand_dims(self.attention_weights, 1), # (batch_size, )
				tf.expand_dims(self.attention_weights_cum, 1)
			), axis=1
		)
		print(f"attention_weights_cat {attention_weights_cat}, shape {attention_weights_cat.shape}")
		self.attention_context, self.attention_weights = self.attention_layer(
			self.attention_hidden, self.memory, self.processed_memory,
			attention_weights_cat, self.mask
		)
		print(f"attention_weights_cat {attention_weights_cat}, shape {attention_weights_cat.shape}")
		print(f"attention_context {self.attention_context}, shape {self.attention_context.shape}")
		print(f"attention_weights {self.attention_weights}, shape {self.attention_weights.shape}")
		exit()

		self.attention_weights_cum += self.attention_weights
		decoder_output = tf.concat(
			(self.attention_hidden, self.attention_context), axis=-1
		)
		self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
			decoder_input, (self.decoder_hidden, self.decoder_cell)
		)
		self.decoder_hidden = self.decoder_dropout(
			self.decoder_hidden, self.training
		)

		decoder_hidden_attention_context = tf.concat(
			(self.decoder_hidden, self.attention_context),
			axis=1
		)
		decoder_output = self.linear_projection(
			decoder_hidden_attention_context
		)

		gate_prediction = self.gate_layer(decoder_hidden_attention_context)
		return decoder_output, gate_prediction, self.attention_weights


	# def call(self, memory, decoder_inputs, memory_lengths):
	def call(self, inputs, training=None):
		self.training = training
		if training:
			# memory (batch_size, max_len, dims) output from encoder.
			# decoder_inputs (batch_size, mel_len, n_mel_channels) the
			#	ground-truth mel-spectrogram
			# memory_lengths (batch_size, text_len) lengths of all text
			#	inputs.
			print(f"In training")
			print(f"inputs length {len(inputs)}")
			memory, decoder_inputs, memory_lengths = inputs
			print(f"raw memory (encoder output) input shape: {memory.shape}")
			print(f"raw decoder (mel-spec) input shape: {decoder_inputs.shape}")
			print(f"raw memory_lengths (text lengths) input shape: {memory_lengths.shape}")

			# decoder_input (no 's'!) (1, batch_size, n_mels_channels *
			#	n_frames_per_step) or (1, batch_size, n_mels_channels) if
			#	n_frames_per_step = 1. It is also a zero tensor.
			decoder_input = tf.expand_dims(
				self.get_go_frame(memory), axis=0
			)
			print(f"decoder input (unsqueezed go frame) shape: {decoder_input.shape}")
			decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
			print(f"decoder inputs shape: {decoder_inputs.shape}")
			decoder_inputs = tf.concat(
				[decoder_input, decoder_inputs], axis=0
			)
			print(f"decoder inputs (concat with decoder_input) shape: {decoder_inputs.shape}")
			decoder_inputs = self.prenet(decoder_inputs, training=training)
			print(f"decoder inputs (prenet output) shape: {decoder_inputs.shape}")

			print("initialize decoder states")
			self.initialize_decoder_states(
				memory, mask=~get_mask_from_lengths(memory_lengths)
			)

			print("loop over decoder inputs")
			mel_outputs, gate_outputs, alignments = [], [], []
			while len(mel_outputs) < tf.shape(decoder_inputs)[0] - 1:
				decoder_input = decoder_inputs[len(mel_outputs)]
				mel_output, gate_output, attention_weights = self.decode(
					decoder_input
				)
				mel_outputs += [tf.squeeze(mel_output, axis=1)]
				gate_outputs += [tf.squeeze(gate_output, axis=1)]
				alignments += [attention_weights]
		else:
			memory = inputs

			decoder_input = self.get_go_frame(memory)

			self.initialize_decoder_states(
				memory, mask=None
			)

			mel_outputs, gate_padded, alignments = [], [], []
			while True:
				decoder_input = self.prenet(decoder_input)
				mel_outputs, gate_padded, alignment = self.decode(
					decoder_input
				)

				mel_outputs += [tf.squeeze(mel_output, axis=1)]
				gate_outputs += [gate_output]
				alignments += [alignment]

				if tf.math.sigmoid(gate_output) > self.gate_threshold:
					break
				elif len(mel_outputs) == self.max_decoder_steps:
					print("Warning! Reached max decoder steps")
					break

				decoder_input = mel_output

		mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
			mel_outputs, gate_outputs, alignments
		)

		return mel_outputs, gate_outputs, alignments 


class Tacotron2(keras.Model):
	def __init__(self, hparams):
		super(Tacotron2, self).__init__()
		self.mask_padding = hparams.mask_padding
		self.n_mel_channels = hparams.n_mel_channels
		self.n_frames_per_step = hparams.n_frames_per_step
		self.embedding = layers.Embedding(
			hparams.n_symbols, hparams.symbols_embedding_dim
		)
		std = sqrt(
			2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim)
		)
		val = sqrt(3.0) * std # uniform bounds for std
		self.encoder = Encoder(hparams)
		self.decoder = Decoder(hparams)
		self.postnet = Postnet(hparams)


	def parse_batch(self, batch):
		text_padded, input_lengths, mel_padded, gate_padded,\
			output_lengths = batch
		text_padded = tf.cast(text_padded, dtype=tf.int64)
		input_lengths = tf.cast(input_lengths, dtype=tf.int64)
		max_len = tf.math.reduce_max(input_lengths).numpy().item()
		mel_padded = tf.cast(mel_padded, dtype=tf.float32)
		gate_padded = tf.cast(gate_padded, dtype=tf.float32)
		output_lengths = tf.cast(output_lengths, dtype=tf.int64)

		return (
			(
				text_padded, input_lengths, mel_padded, max_len, 
				output_lengths
			),
			(mel_padded, gate_padded)
		)


	def parse_output(self, outputs, output_lengths=None):
		if self.mask_padding and output_lengths is not None:
			mask = ~get_mask_from_lengths(output_lengths)
			mask = tf.broadcast_to(
				mask, 
				[
					self.n_mel_channels, tf.shape(mask)[0], 
					tf.shape(mask)[1]
				]
			)
			mask = tf.transpose(mask, (1, 0, 2))

			outputs[0] = tf.where(mask, outputs[0], 0.0)
			outputs[1] = tf.where(mask, outputs[1], 0.0)
			outputs[2] = tf.where(mask[:, 0, :], outputs[2], 1e3) # gate energies

		return outputs


	def _forward(self, inputs):
		text_inputs, text_lengths, mels, max_len,\
			output_lengths = inputs

		embedded_inputs = tf.transpose(
			self.embedding(text_inputs), (1, 2)
		)

		encoder_outputs = self.encoder(embedded_inputs, training=True)

		mel_outputs, gate_outputs, alignments = self.decoder(
			(encoder_outputs, mels, text_lengths) # memory_lengths = text_lengths
		)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[
				mel_outputs, mel_outputs_postnet, gate_outputs,
				alignments
			],
			output_lengths
		)


	def _inference(self, inputs):
		embedded_inputs = tf.transpose(self.embedding, (1, 2))
		encoder_outputs = self.encoder(embedded_inputs, training=False)
		mel_outputs, gate_outputs, alignments = self.decoder(
			encoder_outputs
		)

		mel_outputs_postnet = self.postnet(mel_outputs)
		mel_outputs_postnet = mel_outputs + mel_outputs_postnet

		return self.parse_output(
			[
				mel_outputs, mel_outputs_postnet, gate_outputs,
				alignments
			]
		)


	def call(self, inputs, training=None):
		if training:
			self._forward(inputs)
		else:
			self._inference(inputs)


	def train_step(self, data):
		# Unpack the data from the batch.
		text_inputs, text_len, mels, gate, output_len = data

		# Calculate max_len from text_len.
		max_len = tf.reduce_max(text_len)

		# Set input and outputs for training.
		x = (text_inputs, text_len, mels, max_len, output_len)
		y = (mels, gate)

		with tf.GradientTape() as tape:
			# Pass through Tacotron2 in training mode.
			y_pred = self.call(x, training=True)

			# Compute loss.
			loss = self.loss(y, y_pred)

		# Compute gradients and apply with optimizer.
		grads = tape.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.comiled_metrics.update_state(y, y_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}
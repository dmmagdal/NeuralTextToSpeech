# test_tf_tacotron2.py


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from hparams import HParams
from model import *


def main():
	# Initialize hyperparameters
	hparams = HParams()

	# Text (encoded) text input (of length 141).
	# len(symbols) = 148. That should be the max value for the text
	# encoded inputs.
	fake_text = tf.random.uniform(
		shape=(141,), minval=0, maxval=hparams.n_symbols, 
		dtype=tf.int64
	)
	fake_text = tf.expand_dims(fake_text, 0) # expand for batch size 1
	input_lens = tf.convert_to_tensor([len(fake_text)], dtype=tf.int64)
	print(f"Fake input text shape: {fake_text.shape}")

	# Embedding layer that is a part of the initial tacotron2 model.
	std = sqrt(
		2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim)
	)
	val = sqrt(3.0) * std
	# embedding = nn.Embedding(
	# 	hparams.n_symbols, hparams.symbols_embedding_dim
	# )
	# embedding.weight.data.uniform_(-val, val)

	# Initialize embedding layer. Set the embedding weights (with the
	# embeddings_initializer) through the
	# keras.initializers.RandomUniform(), setting the min and max
	# values calculated.
	embedding = layers.Embedding(
		hparams.n_symbols, hparams.symbols_embedding_dim,
		embeddings_initializer=keras.initializers.RandomUniform(
			minval=-val, maxval=val
		)
	)

	# tacotron2 encoder.
	encoder = Encoder(hparams)

	# Pass dummy text through tacotron embedding and encoder. The
	# expected output shape is to be (batch_size, text_len,
	# hparams.symbols_embedding_dim). Note:
	# hparams.symbols_embedding_dim is expected to be the same value
	# as hparams.encoder_embedding_dim. 
	emb_out = embedding(fake_text)
	print(f"Embedding output shape: {emb_out.shape}")

	# By default, Tensorflow conv (ConvNorm) layers take the channel as
	# the last dimension. In pytorch, this is not the case, so the
	# embedding outputs have to have to be transposed (embedding output
	# is of shape (batch, text_len, embedding_dim), the conv layers
	# expect a shape of (batch, embedding_dim, text_len) where
	# embedding_dim is the channels dim). This can be ignored for
	# tensorflow.
	print("-" * 72)

	# Note: No need to set layers to eval() mode. Tensorflow makes no
	# distinction because a call to layers is the same. If there is
	# specific code to be run during inference, the training parameter
	# to the layer's call() function should help steer the flow of
	# information. Also, the training and inference outputs should be
	# the same shape.

	enc_out = encoder(emb_out, input_lens)
	print(f"Encoder output shape: {enc_out.shape}")
	print("-" * 72)

	# Fake mel, gate, max_len, mel_len inputs. Mel length is going to
	# be 1126. Expand dims for batch size 1.
	fake_mel = tf.random.normal(shape=(1126, hparams.n_mel_channels))
	fake_mel = tf.expand_dims(fake_mel, 0)
	# fake_gates = tf.ones((1126,))
	fake_gates = np.ones((1126,), dtype=np.int_) # tensors are immutable. Initialize with numpy then convert to tensor
	fake_gates[-1] = 0
	fake_gates = tf.expand_dims(
		tf.convert_to_tensor(fake_gates, dtype=tf.int64), 0
	)
	mel_lens = tf.convert_to_tensor([1126], dtype=tf.int64)
	max_len = tf.convert_to_tensor([1126], dtype=tf.int64)
	print(f"Fake input mel shape: {fake_mel.shape}")
	print(f"Fake gate shape: {fake_gates.shape}")
	print(f"Fake mel_lens shape: {mel_lens.shape}")
	print(f"Fake max_len shape: {max_len.shape}")

	# tacotron2 decoder.
	decoder = Decoder(hparams)

	# Transpose mel input because decoder expects mel to be of shape
	# (batch, n_mel_channels, mel_length).
	# dec_out = decoder(enc_out, fake_mel.transpose(1, 2), memory_lengths=input_lens)
	dec_out = decoder((enc_out, fake_mel, input_lens), training=True)
	print(f"Decoder output length: {len(dec_out)}")
	for i in range(len(dec_out)):
		print(F'\tDecoder output {i + 1} shape: {dec_out[i].size()}')
	'''
	decoder.eval()
	dec_inf_out = decoder.inference(enc_inf_out)
	print(f"Decoder (inference) output length: {len(dec_inf_out)}")
	for i in range(len(dec_inf_out)):
		print(F'\tDecoder (inference) output {i + 1} shape: {dec_inf_out[i].size()}')

	# Decoder outputs 3 tensors:
	# 1) mel-spec outputs
	# 2) gate outputs
	# 3) alignments

	print("-" * 72)

	# full tacotron2 model
	tacotron2 = Tacotron2(hparams)

	# tacotron2 outputs (postnet mel)
	tac2_x, tac2_y = tacotron2.parse_batch(
		(
			fake_text, input_lens, fake_mel.transpose(1, 2), 
			fake_gates, mel_lens
		)
	)
	tac2_out = tacotron2(
		tac2_x
	)
	print(f"Tacotron2 output len: {len(tac2_out)}")
	for i in range(len(tac2_out)):
		print(F'\tTacotron2 output {i + 1} shape: {tac2_out[i].size()}')
	print("-" * 32)

	# Tacotron outputs 4 tensors:
	# 1) decoder mel-spec outputs
	# 2) postnet mel-spec outputs
	# 2) gate outputs
	# 3) alignments

	tacotron2.eval()
	tac2_inf_out = tacotron2.inference(fake_text)
	print(f"Tacotron2 (inference) output len: {len(tac2_inf_out)}")
	for i in range(len(tac2_inf_out)):
		print(F'\tTacotron2 (inference) output {i + 1} shape: {tac2_inf_out[i].size()}')
	'''

	# Exit the program
	exit()


if __name__ == '__main__':
	main()
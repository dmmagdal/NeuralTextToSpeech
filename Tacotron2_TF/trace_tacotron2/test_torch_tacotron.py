# tacotron2_flow.py
# Iterate through tacotron work flow to get the idea of the different
# shapes.


from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

# from tacotron2.model import *
from model_tacotron2 import *
from hparams import HParams


def main():
	# Initialize hyperparameters
	hparams = HParams()

	# Test (encoded) text input (of length 141).
	# len(symbols) = 148. That should be the max value for the text
	# encoded inputs.
	fake_text = torch.randint(low=0, high=148, size=(141,))
	fake_text = fake_text.unsqueeze(0) # Expand for batch size 1.
	input_lens = torch.tensor([len(fake_text)])
	print(f"Fake input text shape: {fake_text.size()}")

	# Embedding layer that is a part of the initial tacotron2 model.
	std = sqrt(
		2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim)
	)
	val = sqrt(3.0) * std
	embedding = nn.Embedding(
		hparams.n_symbols, hparams.symbols_embedding_dim
	)
	embedding.weight.data.uniform_(-val, val)

	# tacotron2 encoder.
	encoder = Encoder(hparams)

	# Pass dummy text through tacotron embedding and encoder. The
	# expected output shape is to be (batch_size, text_len,
	# hparams.symbols_embedding_dim). Note:
	# hparams.symbols_embedding_dim is expected to be the same value
	# as hparams.encoder_embedding_dim. 
	emb_out = embedding(fake_text)
	print(f"Embedding output shape: {emb_out.size()}")

	# Transpose embedding output before passing to encoder. Transpose
	# the text_len dimension with the last dim. This is because the
	# ConvNorm layers used in the encoder DO NOT take the channel as
	# the last dimension (whereas Tensorflow does by default).
	emb_out = emb_out.transpose(1, 2)
	print(f"Embedding post transpose shape: {emb_out.size()}")
	print("-" * 72)

	# Note: Be sure to set layers to .eval() mode before calling
	# inference() functions. This is required for pytorch models and
	# does affect the output.

	enc_out = encoder(emb_out, input_lens)
	print(f"Encoder output shape: {enc_out.size()}")
	encoder.eval()
	enc_inf_out = encoder.inference(emb_out)
	print(f"Encoder (inference) output shape: {enc_inf_out.size()}")	
	print("-" * 72)

	# Fake mel, gate, max_len, mel_len inputs. Mel length is going to
	# be 1126. Expand dims for batch size 1.
	fake_mel = torch.randn((1126, hparams.n_mel_channels))
	fake_mel = fake_mel.unsqueeze(0)
	fake_gates = torch.ones((1126,))
	fake_gates[-1] = 0
	fake_gates = fake_gates.unsqueeze(0)
	mel_lens = torch.tensor([1126])
	max_len = torch.tensor([1126])
	print(f"Fake input mel shape: {fake_mel.size()}")
	print(f"Fake gate shape: {fake_gates.size()}")
	print(f"Fake mel_lens shape: {mel_lens.size()}")
	print(f"Fake max_len shape: {max_len.size()}")

	# tacotron2 decoder.
	decoder = Decoder(hparams)

	# Transpose mel input because decoder expects mel to be of shape
	# (batch, n_mel_channels, mel_length).
	dec_out = decoder(enc_out, fake_mel.transpose(1, 2), memory_lengths=input_lens)
	print(f"Decoder output length: {len(dec_out)}")
	for i in range(len(dec_out)):
		print(F'\tDecoder output {i + 1} shape: {dec_out[i].size()}')
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

	# Exit the program
	exit()


if __name__ == '__main__':
	main()
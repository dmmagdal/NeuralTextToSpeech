# inference.py
#
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import argparse
import datetime as dt
import json
import os
import sys
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import params
from model import GradTTS
from utils import intersperse
from text import text_to_sequence, cmudict
from text.symbols import symbols


def main():
	# Import argument parser for model inference arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
	parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
	parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
	parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
	args = parser.parse_args()

	# Parse speaker args.
	if not isinstance(args.speaker_id, type(None)):
		assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
		spk = tf.convert_to_tensor([args.speaker_id], dtype=tf.int64)
	else:
		spk = None

	# Initialize model.
	model = GradTTS(
		params.n_symbols, 1, None, params.n_enc_channels, 
		params.filter_channels, params.filter_channels_dp, 
		params.n_heads, params.n_enc_layers, params.enc_kernel,
		params.enc_dropout, params.window_size, params.n_feats, 
		params.dec_dim, params.beta_min, params.beta_max, 
		params.pe_scale, params.out_size
	)

	# Load trained model weights/state.

	# Initialize vocoder.

	# Load Vocoder model.

	# Load in target text from file.
	with open(args.file, 'r', encoding='utf-8') as f:
		texts = [line.strip() for line in f.readlines()]
	cmu = cmudict.CMUDict('./resources/cmu_dictionary')

	for i, text in enumerate(texts):
		print(f'Synthesizing {i} text...', end=' ')
		x = tf.expand_dims(
			tf.convert_to_tensor(
				intersperse(
					text_to_sequence(text, dictionary=cmu), len(symbols)
				),
				dtype=tf.int64
			), 
			axis=0
		)
		x_lengths = tf.convert_to_tensor([x.shape[-1]], dtype=tf.int64)
		print(f"x {x}, shape {x.shape}")
		print(f"x_lengths {x_lengths}, shape {x_lengths.shape}")

		t = dt.datetime.now()
		# y_enc, y_dec, attn = model.call(
		# 	x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
		# 	stoc=False, spk=spk, length_scale=0.91
		# )
		y_enc, y_dec, attn = model.call({
			"x": x, "x_lengths": x_lengths, 
			"n_timesteps": args.timesteps, "temperature": 1.5, 
			"stoc": False, "spk": spk, "length_scale": 0.91
		})
		t = (dt.datetime.now() - t).total_seconds()
		print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

		# Encode mel-spectrogram through vocoder to wav format.
		# audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

		# Write wav audio to file.
		# write(f'./out/sample_{i}.wav', 22050, audio)

	print('Done. Check out `out` folder for samples.')

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
# data.py
# Classes and functions to load and preprocess dataset.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from common.audio_processing_tf import STFT


def load_filepaths_and_text(filelist, split="|"):
	if isinstance(filelist, str):
		with open(filelist, encoding='utf-8') as f:
			filepaths_and_text = [
				line.strip().split(split) for line in f
			]
	else:
		filepaths_and_text = filelist
	return filepaths_and_text


def load_wav_to_tensorflow(full_path):
	""" Loads wavdata into tensorflow array """
	file = tf.io.read_file(full_path)

	audio, sampling_rate = tf.audio.decode_wav(file)
	audio = tf.squeeze(audio, axis=-1)

	audio = tf.cast(audio, tf.float32)
	return audio, sampling_rate


class Data:
	def __init__(self, dataset_path, filelist_path, 
			n_mel_channels=80, sampling_rate=22050, filter_length=1024,
			hop_length=256, win_length=1024, mel_fmin=0.0,
			mel_fmax=8000, load_mel_from_disk=False, n_speakers=1):
		self.dataset_path = dataset_path
		self.audiopaths_and_text = load_filepaths_and_text(
			filelist_path
		)
		self.filter_length = filter_length # same as n_fft
		self.n_mel_channels = n_mel_channels
		self.sampling_rate = sampling_rate
		self.hop_length = hop_length
		self.win_length = win_length
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		self.max_wav_value = 32768.0
		random.seed(1234)
		random.shuffle(self.audiopaths_and_text)

		self.stft = STFT(
			filter_length=filter_length, frame_step=hop_length,
			frame_length=win_length, sampling_rate=sampling_rate,
			mel_fmin=mel_fmin, mel_fmax=mel_fmax
		)
		self.load_mel_from_disk = load_mel_from_disk
		self.n_speakers = n_speakers

		os.makedirs(self.dataset_path, exist_ok=True)

		self.max_target_len = -1


	def __getitem__(self, index):
		# Separate filename and text
		if self.n_speakers > 1:
			audiopath, _, _ = self.audiopaths_and_text[index]
		else:
			audiopath, _ = self.audiopaths_and_text[index]

		mel = self.get_mel(audiopath)

		# return (
		# 	text, mel, len(text), pitch, energy, speaker, attn_prior,
		# 	audiopath
		# )
		return {"mel": mel}


	def __len__(self):
		return len(self.audiopaths_and_text)


	def get_mel(self, filename):
		# Check that file exists.
		base_file = os.path.basename(filename)
		saved_mel = os.path.join(
			self.dataset_path, 
			base_file.replace(".wav", "_mel.npy")
		)

		if not self.load_mel_from_disk or not os.path.exists(saved_mel):
			audio, sampling_rate = load_wav_to_tensorflow(filename)
			if sampling_rate != self.stft.sampling_rate:
				raise ValueError(
					"{} SR doesn't match target {} SR".format(
						sampling_rate, self.stft.sampling_rate
					)
				)
			audio_norm = audio / self.max_wav_value
			audio_norm = tf.expand_dims(audio_norm, 0)
			melspec = self.stft.mel_spectrogram(audio_norm)
			np.save(saved_mel, melspec.numpy())
		else:
			melspec = tf.convert_to_tensor(np.load(saved_mel))
		return melspec


	def get_max_lengths(self):
		# Compute the maximum target (mel-spectrogram) lengths.
		print("Isolating max target lengths...")
		for idx in tqdm(range(len(self.audiopaths_and_text))):
			# Use this (re-uses the code at the beginning of
			# __getitem__), instead of the __getitem__ function to
			# reduce overhead from computing/loading all other
			# unnecessary values. Only the mel spectrograms are needed
			# to find the maximum input and output lengths. Current
			# time is down to less than 10 minutes vs hours when using
			# __getitem__.
			if self.n_speakers > 1:
				audiopath, _, _ = self.audiopaths_and_text[idx]
			else:
				audiopath, _ = self.audiopaths_and_text[idx]

			mel = self.get_mel(audiopath)
			self.max_target_len = max(
				tf.shape(mel)[0], self.max_target_len
			)

		print(f"Max target length {self.max_target_len}")


	def generator(self):
		# Compute the maximum target (mel-spectrogram) lengths.
		if (self.max_target_len < 0):
			self.get_max_lengths()

		assert self.max_target_len != -1

		print(f"Max target length {self.max_target_len}")

		# Apply data collate function to each item in the dataset.
		print("Applying data collator function...")
		for idx in tqdm(range(len(self.audiopaths_and_text))):
			yield self.collate_fn(self.__getitem__(idx))


	def tensor_slices(self):
		# Compute the maximum target (mel-spectrogram) lengths.
		if (self.max_target_len < 0):
			self.get_max_lengths()

		assert self.max_target_len != -1

		print(f"Max target length {self.max_target_len}")

		# Apply data collate function to each item in the dataset.
		print("Applying data collator function...")
		tensor_list = []
		for idx in tqdm(range(len(self.audiopaths_and_text))):
			tensor_list.append(self.collate_fn(self.__getitem__(idx)))
		return tensor_list


	def collate_fn(self, batch):
		# Unpack the batch tuple.
		# (
		# 	text_encoded, mel, text_len, pitch, energy, speaker_id, 
		# 	attn_prior, audiopath
		# ) = batch
		mel = batch["mel"]

		# The following assertions make sure that the text and mel
		# spectrogram lengths do not exceed the maximums set in the
		# object.
		assert self.max_target_len >= mel.shape[0], f"Target mel specrogram length exceeds maximum length ({self.max_target_len}): Received {mel.shape[0]}"
		
		# Right zero-pad mel-spec.
		n_mel_channels = mel.shape[1]

		# Mel padded
		mel_padded = np.zeros(
			(self.max_target_len, n_mel_channels), dtype=np.float32
		)
		mel_padded[:mel.shape[0], :] = mel
		mel_padded = tf.convert_to_tensor(mel_padded)

		# Convert remaining values to tensors (input lengths, output
		# lengths, speaker id, and audiopath).
		output_lengths = tf.convert_to_tensor(
			mel.shape[0], dtype=tf.int64
		)

		# Outputs are the following:
		# -> padded encoded texts (dtype=tf.int64, 
		#	shape=(max_input_length))
		# -> input lengths (dtype=tf.int64, shape=())
		# -> padded mel spectrograms (dtype=tf.float32,
		#	shape=(max_target_length, n_mel_channels))
		# -> output lengths (dtype=tf.int64, shape=())
		# -> len_x (dtype=tf.int64, shape=())
		# -> padded pitch (dtype=tf.float32, 
		#	shape=(n_formants, max_target_lengths + 4))
		# -> padded energy (dtype=tf.float32,
		#	shape=(max_target_length,))
		# -> speaker id (dtype=tf.int64, shape=())
		# -> padded attention priors (dtype=tf.float32,
		#	shape=(max_target_length, max_input_length))
		# -> audiopath (dtype=tf.string, shape=())
		# return (
		# 	text_padded, input_lengths, mel_padded, output_lengths,
		# 	len_x, pitch_padded, energy_padded, speaker_id, 
		# 	attn_prior_padded, audiopath
		# )
		# return {
		# 	"text": text_padded, "text_length": input_lengths,
		# 	"mel": mel_padded, "mel_length": output_lengths,
		# 	"spkr": speaker
		# }
		return (
			mel_padded, output_lengths, 
		)
# data.py
# Classes and functions to load and preprocess dataset.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import math
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
	def __init__(self, filelist_path, segment_size, filter_length=1024,
			n_mel_channels=80, hop_length=256, win_length=1024, 
			sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000, 
			split=True, shuffle=True, mel_fmax_loss=None, 
			fine_tuning=False, base_mels_path=None, n_speakers=1):
		# File locations.
		self.audiopaths_and_text = load_filepaths_and_text(
			filelist_path
		)
		self.base_mels_path = base_mels_path

		# STFT.
		self.filter_length = filter_length # same as n_fft
		self.n_mel_channels = n_mel_channels
		self.sampling_rate = sampling_rate
		self.hop_length = hop_length
		self.win_length = win_length
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		self.mel_fmax_loss = mel_fmax_loss # replaces mel_fmax for mel_loss

		self.stft = STFT(
			filter_length=filter_length, frame_step=hop_length,
			frame_length=win_length, sampling_rate=sampling_rate,
			mel_fmin=mel_fmin, mel_fmax=mel_fmax
		)
		self.stft_loss = STFT(
			filter_length=filter_length, frame_step=hop_length,
			frame_length=win_length, sampling_rate=sampling_rate,
			mel_fmin=mel_fmin, 
			mel_fmax=mel_fmax_loss if mel_fmax_loss else mel_fmax
		) # added for mel_loss

		# Randomness.
		random.seed(1234)
		if shuffle:
			random.shuffle(self.audiopaths_and_text)

		# Audio loading.
		self.segment_size = segment_size
		self.n_speakers = n_speakers
		self.max_wav_value = 32768.0

		# Fine-tuning vs training & split.
		self.fine_tuning = fine_tuning
		self.split = split


	def __getitem__(self, index):
		# Separate filename and text
		if self.n_speakers > 1:
			audiopath, _, _ = self.audiopaths_and_text[index]
		else:
			audiopath, _ = self.audiopaths_and_text[index]

		return self.get_mel_audio(audiopath)


	def __len__(self):
		return len(self.audiopaths_and_text)


	def get_mel_audio(self, filename):
		# Load audio from wav.
		audio, sampling_rate = load_wav_to_tensorflow(filename)
		if sampling_rate != self.stft.sampling_rate:
			raise ValueError(
				"{} SR doesn't match target {} SR".format(
					sampling_rate, self.stft.sampling_rate
				)
			)
		audio_norm = audio / self.max_wav_value
		audio_norm = tf.expand_dims(audio_norm, 0)
		audio = audio_norm

		# Different behavior for fine-tuning vs training from scratch.
		if not self.fine_tuning:
			if self.split:
				if audio.shape[1] >= self.segment_size:
					# Subsample the audio if it is too large (larger
					# than segment_size) into a piece that is
					# segment_size.
					max_audio_start = audio.shape[1] - self.segment_size
					audio_start = random.randint(0, max_audio_start)
					audio = audio.numpy()[
						:, audio_start:audio_start + self.segment_size
					]
				else:
					# Pad the audio if it is too small (shorter than
					# segment_size) to segment_size.
					audio = tf.pad(
						audio, 
						[[0, self.segment_size - audio.shape[1]]], 
						"CONSTANT"
					)

				# I added this to convert the audio back to tensorflow
				# tensor.
				audio = tf.convert_to_tensor(audio)
			
			# Extract the audio mel-spectrogram.
			mel = self.stft.mel_spectrogram(audio)
		else:
			# Load mel-spectrogram processed from Tacotron2 teacher
			# forcing.
			mel = np.load(
				os.path.join(
					self.base_mels_path,
					os.path.splitext(os.path.split(filename)[-1])[0]
				) + '.npy'
			)
			mel = tf.convert_to_tensor(mel)

			# Expand dims if number if dims < 3 (I dont know why this
			# is done in the original repo).
			if len(mel.shape) < 3:
				mel = tf.expand_dims(mel, 0)

			if self.split:
				frames_per_seg = math.ceil(
					self.segment_size / self.hop_length
				)

				if audio.shape[0] >= self.segment_size:
					# Subsample the audio if it is too large (larger 
					# than segment_size) into a piece that is 
					# segment_size.

					# Note: The orignal implementation has mel.size(2)
					# but I use mel.shape[1]. This is because I assume 
					# in the original, the mel length dims are
					# different. The README.md in the original repo
					# references Tacotron 2, specificialy the NVIDIA
					# repo implementation which is in pytorch. 
					# Mel-spectrograms in pytorch are usually shaped
					# (n_mel_channels, mel_len) while I usually have 
					# them shaped (mel_len, n_mel_channels) in\
					# tensorflow.
					mel_start = random.randint(
						0, mel.shape[1] - frames_per_seg - 1
					)
					mel = mel.numpy()[
						:, mel_start:mel_start + frames_per_seg, :
					]
					audio = audio.numpy()[
						:, mel_start * self.hop_length:(mel_start + frames_per_seg) *  self.hop_length
					]
				else:
					# Pad the audio if it is too small (shorter than
					# segment_size) to segment_size. Do the same for
					# the mel-spectrogram.
					mel = tf.pad(
						mel, 
						[[0, 0], [0, frames_per_seg - mel.shape[1]], [0, 0]],
						"CONSTANT"
					)
					audio = tf.pad(
						audio, 
						[[0, self.segment_size - audio.shape[1]]], 
						"CONSTANT"
					)

				# I added this to convert the audio back to tensorflow
				# tensor.
				audio = tf.convert_to_tensor(audio)

		# Not sure what mel loss is when you consider the audio
		# (perhaps it really matters when there is a difference between
		# the mel-spectrograms which may only apply in fine-tuning).
		mel_loss = self.stft_loss.mel_spectrogram(audio)

		return (
			tf.squeeze(mel) if mel.shape == 3 else mel,
			tf.squeeze(audio),
			filename,
			tf.squeeze(mel_loss) if mel_loss.shape == 3 else mel_loss,
		)


	def generator(self):
		# Extract the mel-spectrogram and audio data.
		for idx in tqdm(range(len(self.audiopaths_and_text))):
			yield self.__getitem__(idx)


class DataOld:
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
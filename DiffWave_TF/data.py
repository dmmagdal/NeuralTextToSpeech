# data.py
# Classes and functions to load and preprocess dataset.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from common.audio_processing_tf import STFT


random.seed(1)


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
	def __init__(self, dataset_path, filelist_path, params,
			n_mel_channels=80, sampling_rate=22050, filter_length=1024,
			hop_length=256, win_length=1024, mel_fmin=0.0,
			mel_fmax=8000, load_mel_from_disk=False, n_speakers=1,
			from_gtzan=False):
		# Data paths.
		self.dataset_path = dataset_path
		self.audiopaths_and_text = load_filepaths_and_text(
			filelist_path
		)
		os.makedirs(self.dataset_path, exist_ok=True)
		
		self.params = params
		self.from_gtzan = from_gtzan # regular or gtzan
		
		# Mel spectrogram parameters.
		self.filter_length = filter_length # same as n_fft
		self.n_mel_channels = n_mel_channels
		self.sampling_rate = sampling_rate
		self.hop_length = hop_length
		self.win_length = win_length
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax
		self.max_wav_value = 32768.0

		# STFT.
		self.stft = STFT(
			filter_length=filter_length, frame_step=hop_length,
			frame_length=win_length, sampling_rate=sampling_rate,
			mel_fmin=mel_fmin, mel_fmax=mel_fmax
		)
		self.load_mel_from_disk = load_mel_from_disk
		self.n_speakers = n_speakers


	def __getitem__(self, index):
		# Separate filename and text
		if self.n_speakers > 1:
			audiopath, text, speaker = self.audiopaths_and_text[index]
			speaker = int(speaker)
		else:
			audiopath, text = self.audiopaths_and_text[index]
			speaker = 1 #None
		# speaker = tf.convert_to_tensor([speaker], dtype=tf.int64)
		speaker = tf.convert_to_tensor(speaker, dtype=tf.int64)

		audio, mel = self.get_mel_audio(audiopath)
	
		return audio, mel


	def __len__(self):
		return len(self.audiopaths_and_text)


	def get_mel_audio(self, filename):
		# Load audio.
		audio, sampling_rate = load_wav_to_tensorflow(filename)
		if sampling_rate != self.stft.sampling_rate:
			raise ValueError(
				"{} SR doesn't match target {} SR".format(
					sampling_rate, self.stft.sampling_rate
				)
			)
		audio_norm = audio / self.max_wav_value
		audio_norm = tf.expand_dims(audio_norm, 0)

		# For unconditional dataset, there is no mel.
		if self.params.unconditional:
			return audio_norm, None

		# Check that mel npy file exists.
		base_file = os.path.basename(filename)
		saved_mel = os.path.join(
			self.dataset_path, 
			base_file.replace(".wav", "_mel.npy")
		)

		if not self.load_mel_from_disk or not os.path.exists(saved_mel):
			melspec = self.stft.mel_spectrogram(audio_norm)
			# print(f"melspec:\n{melspec}")
			np.save(saved_mel, melspec.numpy())
		else:
			melspec = tf.convert_to_tensor(np.load(saved_mel))
		return audio_norm, melspec


	def generator(self):
		# Apply data collate function to each item in the dataset.
		for idx in tqdm(range(len(self.audiopaths_and_text))):
			audio, mel = self.__getitem__(idx)

			# This was a part of the regular collator in the original
			# implementation but I removed it due to fears of issues
			# with batching. Skip (audio) entries that are too short.
			if not self.from_gtzan and tf.shape(audio)[1] < self.params.audio_len:
				continue
			else:
				yield self.collate_fn((audio, mel))


	def collate_fn(self, batch):
		if self.from_gtzan:
			return self.gtzan_collate(batch)
		else:
			return self.path_collate(batch)
		

	def path_collate(self, batch):
		audio, mel = batch
		samples_per_frame = self.hop_length

		audio = tf.squeeze(audio, 0)

		if self.params.unconditional:
			# Subsample audio only for unconditional dataset.
			start = random.randint(
				0, tf.shape(audio)[-1] - self.params.audio_len
			)
			end = start + self.params.audio_len
			audio_slice = np.zeros((tf.shape(audio)), dtype=np.float32)
			audio_slice = audio[start:end]
			audio_slice = np.pad(
				audio_slice, [0, (end - start) - len(audio_slice)], 
				mode="constant"
			)

			# Convert back to tensorflow tensor(s).
			audio_slice = tf.convert_to_tensor(
				audio_slice, dtype=tf.float32
			)

			return audio_slice, None
		else:
			# Subsample audio and mel spectrogram for conditional
			# dataset.
			start = random.randint(
				0, tf.shape(mel)[0] - self.params.crop_mel_frames
			)
			end = start + self.params.crop_mel_frames
			mel_slice = np.zeros(
				(end - start, tf.shape(mel)[1]), dtype=np.float32
			)
			mel_slice = mel[start:end]
			
			start *= samples_per_frame
			end *= samples_per_frame
			audio_slice = np.zeros(
				(end - start), dtype=np.float32
			)
			audio_slice = audio[start:end]
			audio_slice = np.pad(
				audio_slice, [0, (end - start) - len(audio_slice)], 
				mode="constant"
			)

			# Convert back to tensorflow tensor(s).
			audio_slice = tf.convert_to_tensor(
				audio_slice, dtype=tf.float32
			)
			mel_slice = tf.convert_to_tensor(
				mel_slice, dtype=tf.float32
			)

			return audio_slice, mel_slice


	def gtzan_collate(self, batch):
		audio, mel = batch
		mean_audio_len = self.params.audio_len # change to fit in gpu memory

		# Not sure on input shape of audio. 
		# audio = tf.squeeze(audio, 0)

		# audio total generated time = audio_len * sample_rate
		# GTZAN statistics
		# max len audio 675808; min len audio sample 660000; mean len 
		# audio sample 662117 max audio sample 1; min audio sample -1;
		# mean audio sample -0.0010 (normalized) sample rate of all is
		# 22050.
		
		if audio.shape[-1] < mean_audio_len:
			# Pad.
			audio = tf.pad(
				audio, 
				[[0, 0], [0, mean_audio_len - audio.shape[-1]]], 
				mode="CONSTANT", constant_values=0
			)
		elif audio.shape[-1] > mean_audio_len:
			# Crop.
			start = random.randint(0, audio.shape[-1] - mean_audio_len)
			end = start + mean_audio_len
			audio = audio[:, start:end]
		else:
			audio = audio

		return audio, None
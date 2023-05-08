# mel_dataset.py


import math
import os
import random
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read


MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
	""" Loads wavdata into tensorflow array """
	file = tf.io.read_file(full_path)

	audio, sampling_rate = tf.audio.decode_wav(file)
	audio = tf.squeeze(audio, axis=-1)

	audio = tf.cast(audio, tf.float32)
	return audio, sampling_rate


def load_wav_scipy(full_path):
	""" Loads wav data into tensorflow array using scipy """
	sampling_rate, audio = read(full_path)

	audio = tf.convert_to_tensor(audio, tf.float32)
	return audio, sampling_rate


def dynamic_range_compression_tf(x, C=1, clip_val=1e-5):
	return tf.math.log(
		tf.clip_by_value(x, clip_val, tf.float32.max) * C
	)


def dynamic_range_decompression_tf(x, C=1):
	return tf.math.exp(x) / C


def spectral_normalize_tf(magnitudes):
	output = dynamic_range_compression_tf(magnitudes)
	return output


def spectral_de_normalize_tf(magnitudes):
	output = dynamic_range_decompression_tf(magnitudes)
	return output


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, 
		    win_size, fmin, fmax, center=False):
	if tf.math.reduce_min(y) < -1.0:
		print('min value is ', tf.math.reduce_min(y))
	if tf.math.reduce_max(y) > 1.0:
		print('max value is ', tf.math.reduce_max(y))

	global mel_basis, hann_window
	if fmax not in mel_basis:
		mel = librosa_mel_fn(
			sampling_rate, n_fft, num_mels, fmin, fmax
		)
		mel_basis[str(fmax)+'_'+str(y.device)] = tf.convert_to_tensor(
			mel, dtype=tf.float32
		)
		hann_window[str(y.device)] = tf.signal.hann_window(win_size)

	y = tf.pad(
		tf.expand_dims(y, 1), 
		[[int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)]], 
		mode='REFLECT'
	)
	y = tf.squeeze(y, 1)

	spec = tf.signal.stft(
		y, n_fft, hop_length=hop_size, win_length=win_size, 
		window=hann_window[str(y.device)], center=center, 
		pad_mode='reflect', normalized=False, onesided=True
	)

	spec = tf.math.sqrt(spec.pow(2).sum(-1)+(1e-9))

	spec = tf.linalg.matmul(
		mel_basis[str(fmax)+'_'+str(y.device)], spec
	)
	spec = spectral_normalize_tf(spec)

	return spec


class MelDataset:
	def __init__(self, training_files, segment_size, n_fft, num_mels,
				hop_size, win_size, sampling_rate,  fmin, fmax, 
				split=True, shuffle=True, n_cache_reuse=1, 
				fmax_loss=None, fine_tuning=False, base_mels_path=None
			):
		self.audio_files = training_files
		random.seed(1234)
		if shuffle:
			random.shuffle(self.audio_files)
		self.segment_size = segment_size
		self.sampling_rate = sampling_rate
		self.split = split
		self.n_fft = n_fft
		self.num_mels = num_mels
		self.hop_size = hop_size
		self.win_size = win_size
		self.fmin = fmin
		self.fmax = fmax
		self.fmax_loss = fmax_loss
		self.cached_wav = None
		self.n_cache_reuse = n_cache_reuse
		self._cache_ref_count = 0
		self.fine_tuning = fine_tuning
		self.base_mels_path = base_mels_path


	def __getitem__(self, index):
		filename = self.audio_files[index]
		if self._cache_ref_count == 0:
			audio, sampling_rate = load_wav(filename)
			audio = audio / MAX_WAV_VALUE
			if not self.fine_tuning:
				audio = normalize(audio) * 0.95
			self.cached_wav = audio
			if sampling_rate != self.sampling_rate:
				raise ValueError(
					"{} SR doesn't match target {} SR".format(
						sampling_rate, self.sampling_rate
					)
				)
			self._cache_ref_count = self.n_cache_reuse
		else:
			audio = self.cached_wav
			self._cache_ref_count -= 1

		audio = tf.convert_to_tensor(audio, dtype=tf.float32)
		audio = tf.expand_dims(audio, 0)

		if not self.fine_tuning:
			if self.split:
				if audio.shape[1] >= self.segment_size:
					max_audio_start = audio.shape[1] - self.segment_size
					audio_start = random.randint(0, max_audio_start)
					audio = audio[:, audio_start:audio_start + self.segment_size]
				else:
					audio = tf.pad(
						audio, [[0, self.segment_size - audio.shape[1]]], 
						'CONSTANT'
					)

			mel = mel_spectrogram(
				audio, self.n_fft, self.num_mels, self.sampling_rate, 
				self.hop_size, self.win_size, self.fmin, self.fmax, 
				center=False
			)
		else:
			mel = np.load(
				os.path.join(
					self.base_mels_path, 
					os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'
				)
			)
			mel = tf.convert_to_tensor(mel)

			if len(mel.shape) < 3:
				mel = tf.expand_dims(mel, 0)

			if self.split:
				frames_per_seg = math.ceil(
					self.segment_size / self.hop_size
				)

				if audio.shape[1] >= self.segment_size:
					mel_start = random.randint(
						0, mel.shape[2] - frames_per_seg - 1
					)
					mel = mel[:, :, mel_start:mel_start + frames_per_seg]
					audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
				else:
					mel = tf.pad(
						mel, [[0, frames_per_seg - mel.shape[2]]], 
						'CONSTANT'
					)
					audio = tf.pad(
						audio, 
						[[0, self.segment_size - audio.shape[1]]], 
						'CONSTANT'
					)
		
		mel_loss = mel_spectrogram(
			audio, self.n_fft, self.num_mels, self.sampling_rate, 
			self.hop_size, self.win_size, self.fmin, self.fmax_loss,
			center=False
		)

		return (tf.squeeze(mel), tf.squeeze(audio, 0), filename, tf.squeeze(mel_loss))

	def __len__(self):
		return len(self.audio_files)
# data_utils.py


import random
import numpy as np
import tensorflow as tf
import layers
from utils import load_wav_to_tf, load_filespaths_and_text
from text import text_to_sequence


class TextMelLoader:
	self.audiopaths_and_text = load_filepaths_and_text(
			audiopaths_and_text
		)
		self.text_cleaners = hparams.text_cleaners
		self.max_wav_value = hparams.max_wav_value
		self.sampling_rate = hparams.sampling_rate
		self.load_mel_from_disk = hparams.load_mel_from_disk
		self.stft = layers.TacotronSTFT(
			hparams.filter_length, hparams.hop_length, 
			hparams.win_length, hparams.n_mel_channels, 
			hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax
		)
		random.seed(hparams.seed)
		random.shuffle(self.audiopaths_and_text)


	def get_mel_text_pair(self, audiopaths_and_text):
		# separate filename and text
		audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
		text = self.get_text(text)
		mel = self.get_mel(audiopath)
		return (text, mel)


	def get_mel(self, filename):
		if not self.load_mel_from_disk:
			audio, sampling_rate = load_wav_to_torch(filename)
			if sampling_rate != self.stft.sampling_rate:
				raise ValueError(
					"{} {} SR doesn't match target {} SR".format(
					sampling_rate, self.stft.sampling_rate)
				)
			audio_norm = audio / self.max_wav_value
			# audio_norm = audio_norm.unsqueeze(0)
			# audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
			audio_norm = tf.expand_dims(audio_norm, 0)
			melspec = self.stft.mel_spectrogram(audio_norm)
			# melspec = torch.squeeze(melspec, 0)
			melspec = tf.squeeze(melspec, 0)
		else:
			# melspec = torch.from_numpy(np.load(filename))
			melspec = tf.convert_to_tensor(np.load(filename))
			# assert melspec.size(0) == self.stft.n_mel_channels, (
			# 	'Mel dimension mismatch: given {}, expected {}'.format(
			# 		melspec.size(0), self.stft.n_mel_channels))
			assert melspec.shape[0] == self.stft.n_mel_channels, (
				'Mel dimension mismatch: given {}, expected {}'.format(
					melspec.shape[0], self.stft.n_mel_channels))

		return melspec


	def get_text(self, text):
		# text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
		text_norm = tf.convert_to_tensor(
			text_to_sequence(text, self.text_cleaners), dtype=tf.int32
		)
		return text_norm


	def __getitem__(self, index):
		return self.get_mel_text_pair(self.audiopaths_and_text[index])


	def __len__(self):
		return len(self.audiopaths_and_text)
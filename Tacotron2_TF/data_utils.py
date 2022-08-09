# data_utils.py


import random
import numpy as np
import tensorflow as tf
# import layers
from utils import load_wav_to_tensorflow, load_filepaths_and_text
from text import text_to_sequence
from audio_processing_tf import STFT
from tqdm import tqdm


class TextMelLoader:
	def __init__(self, audiopaths_and_text, hparams):
		self.audiopath_and_text = load_filepaths_and_text(
			audiopaths_and_text
		)
		self.text_cleaners = hparams.text_cleaners
		self.max_wav_value = hparams.max_wav_value
		self.sampling_rate = hparams.sampling_rate
		self.load_mel_from_disk = hparams.load_mel_from_disk
		'''
		self.stft = layers.TacotronSTFT(
			hparams.filter_length, hparams.hop_length, 
			hparams.win_length, hparams.n_mel_channels, 
			hparams.sampling_rate, hparams.mel_fmin, hparams.mel_fmax
		)
		'''
		self.stft = STFT(
			hparams.filter_length, hparams.win_length, 
			hparams.hop_length, hparams.n_mel_channels, 
			hparams.sampling_rate, hparams.mel_fmin, 
			hparams.mel_fmax
		)
		random.seed(hparams.seed)
		random.shuffle(self.audiopath_and_text)


	def get_mel_text_pair(self, audiopath_and_text):
		# separate filename and text
		audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
		text = self.get_text(text)
		mel = self.get_mel(audiopath)
		return text, mel


	def get_mel(self, filename):
		if not self.load_mel_from_disk:
			# audio, sampling_rate = load_wav_to_torch(filename)
			audio, sampling_rate = load_wav_to_tensorflow(filename)
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


	def set_collate_fn(self, collate_fn):
		self.collate_fn = collate_fn


	def generator(self):
		# Compute the maximum input (text) and target (mel-spectrogram)
		# lengths.
		print("Isolating maximum input and target lengths...")
		self.max_input_len = 0
		self.max_target_len = 0
		for idx in tqdm(range(len(self.audiopath_and_text))):
			text, mel = self.__getitem__(idx)
			self.max_input_len = max(
				tf.shape(text)[0], self.max_input_len
			)
			self.max_target_len = max(
				tf.shape(mel)[0], self.max_target_len
			)

		# Update the collate function max lengths. Must have the 
		# collate function initialized before.
		self.collate_fn.update_max_len(
			self.max_input_len, self.max_target_len
		)

		# Apply data collate function to each item in the dataset.
		assert hasattr(self, "collate_fn"), "Collate function for dataset not set."
		print("Applying collator function...")
		for idx in tqdm(range(len(self.audiopath_and_text))):
			text, mel = self.__getitem__(idx)
			yield self.collate_fn(text, mel)


	def __getitem__(self, index):
		return self.get_mel_text_pair(self.audiopath_and_text[index])


	def __len__(self):
		return len(self.audiopath_and_text)



class TextMelCollate:
	def __init__(self, n_frames_per_step):
		self.n_frames_per_step = n_frames_per_step
		self.max_input_len = 0
		self.max_target_len = 0


	def __call__(self, text, mel):
		# The following assertions make sure that the text and mel
		# spectrogram lengths do not exceed the maximums set in the
		# object.
		assert self.max_input_len >= text.shape[0], f"Input encoded text length exceeds maximum length ({self.max_input_len}): Received {text.shape[0]}"
		assert self.max_target_len >= mel.shape[0], f"Target mel specrogram length exceeds maximum length ({self.max_target_len}): Received {mel.shape[0]}"

		# Right zero-pad text.
		text_padded = np.zeros((self.max_input_len,), dtype=np.uint64)
		text_padded[:text.shape[0]] = text
		text_padded = tf.convert_to_tensor(text_padded, dtype=tf.int64)

		# Right zero-pad mel-spectrogram.
		mel_length = tf.shape(mel)[0]
		num_mels = tf.shape(mel)[1]
		if self.max_target_len % self.n_frames_per_step != 0:
			self.max_target_len += self.n_frames_per_step - self.max_target_len % self.n_frames_per_step
			assert self.max_target_len % self.n_frames_per_step == 0

		mel_padded = np.zeros(
			(self.max_target_len, num_mels), dtype=np.float32
		)
		mel_padded[:mel_length, :] = mel
		mel_padded = tf.convert_to_tensor(mel_padded, dtype=tf.float32)
		gate_padded = np.zeros(
			(self.max_target_len,), dtype=np.float32
		)
		gate_padded[mel_length - 1:] = 1
		gate_padded = tf.convert_to_tensor(
			gate_padded, dtype=tf.float32
		)

		# Lengths are already tensors. Convert their dtype to their
		# respective values.
		input_length = tf.cast(tf.shape(text)[0], dtype=tf.int64)
		output_length = tf.cast(mel_length, dtype=tf.int64)

		# Outputs are text padded, input length, mel padded,
		# gate padded, and output length.
		# Output dtypes are tf.int64, tf.int64, tf.float32, tf.float32,
		# tf.int64.
		# Output shapes are (None,), (), (None, n_mel_channels),
		# (None,), ().
		return text_padded, input_length, mel_padded, gate_padded, \
			output_length


	def update_max_len(self, max_input_len, max_target_len):
		# Update the maximum input ((encoded) text) and target/output
		# (mel-spectrogram) lengths.
		self.max_input_len = max_input_len
		self.max_target_len = max_target_len
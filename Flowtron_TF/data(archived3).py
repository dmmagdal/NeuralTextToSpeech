# data.py
# Classes and functions to load and preprocess dataset.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import re
import os
import argparse
import json
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.stats import betabinom
from audio_processing_tf import STFT
from text import text_to_sequence, cmudict, _clean_text, get_arpabet


def beta_binomial_prior_distribution(phoneme_count, mel_count,
		scaling_factor=1.0):
	P, M = phoneme_count, mel_count
	x = np.arange(0, P)
	mel_text_probs = []
	for i in range(1, M+1):
		a, b = scaling_factor*i, scaling_factor*(M+1-i)
		rv = betabinom(P, a, b)
		mel_i_prob = rv.pmf(x)
		mel_text_probs.append(mel_i_prob)
	return tf.convert_to_tensor(np.array(mel_text_probs))


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
	def __init__(self, filelist_path, filter_length, hop_length, 
			win_length, sampling_rate, mel_fmin, mel_fmax, 
			max_wav_value, p_arpabet, cmudict_path, text_cleaners, 
			speaker_ids=None, use_attn_prior=False, 
			attn_prior_threshold=1e-4, prior_cache_path="", 
			betab_scaling_factor=1.0, randomize=True, 
			keep_ambiguous=False, seed=1234):
		self.max_wav_value = max_wav_value
		self.audiopaths_and_text = load_filepaths_and_text(filelist_path)
		self.use_attn_prior = use_attn_prior
		self.betab_scaling_factor = betab_scaling_factor
		self.attn_prior_threshold = attn_prior_threshold
		self.keep_ambiguous = keep_ambiguous

		if speaker_ids is None or speaker_ids == '':
			self.speaker_ids = self.create_speaker_lookup_table(
				self.audiopaths_and_text)
		else:
			self.speaker_ids = speaker_ids

		self.stft = STFT(
			filter_length=filter_length, frame_step=hop_length,
			frame_length=win_length, sampling_rate=sampling_rate,
			mel_fmin=mel_fmin, mel_fmax=mel_fmax
		)
		self.sampling_rate = sampling_rate
		self.text_cleaners = text_cleaners
		self.p_arpabet = p_arpabet
		self.cmudict = cmudict.CMUDict(
			cmudict_path, keep_ambiguous=keep_ambiguous)
		if speaker_ids is None:
			self.speaker_ids = self.create_speaker_lookup_table(
				self.audiopaths_and_text)
		else:
			self.speaker_ids = speaker_ids

		# caching makes sense for p_phoneme=1.0
		# for other values, everytime text lengths will change
		self.prior_cache_path = prior_cache_path
		self.caching_enabled = False
		if (self.prior_cache_path is not None and
				self.prior_cache_path != "" and p_arpabet == 1.0):
			self.caching_enabled = True
		# make sure caching path exists
		if (self.caching_enabled and
				not os.path.exists(self.prior_cache_path)):
			os.makedirs(self.prior_cache_path)

		self.max_input_len = 0
		self.max_target_len = 0

		random.seed(seed)
		if randomize:
			random.shuffle(self.audiopaths_and_text)


	def compute_attention_prior(self, audiopath, mel_length, text_length):
		folder_path = audiopath.split('/')[-2]
		filename = os.path.basename(audiopath).split('.')[0]
		prior_path = os.path.join(
			self.prior_cache_path, folder_path + "_" + filename
		)

		# prior_path += "_prior.pth"
		prior_path += "_prior.npy"

		prior_loaded = False
		if self.caching_enabled and os.path.exists(prior_path):
			# TODO: Replace torch.load()
			# attn_prior = torch.load(prior_path)
			attn_prior = tf.convert_to_tensor(np.load(prior_path))
			if (attn_prior.shape[1] == text_length and
					attn_prior.shape[0] == mel_length):
				prior_loaded = True
			else:
				print("Prior size mismatch, recomputing")

		if not prior_loaded:
			attn_prior = beta_binomial_prior_distribution(
				text_length, mel_length, self.betab_scaling_factor
			)
			# TODO: Replace torch.save()
			if self.caching_enabled:
				# torch.save(attn_prior, prior_path)
				np.save(prior_path, attn_prior.numpy())

		if self.attn_prior_threshold > 0:
			# attn_prior = attn_prior.masked_fill(
			# 	attn_prior < self.attn_prior_threshold, 0.0)
			attn_prior = tf.where(
				attn_prior < self.attn_prior_threshold, 0.0, attn_prior
			)
		# print(attn_prior.shape)

		return attn_prior


	def create_speaker_lookup_table(self, audiopaths_and_text):
		speaker_ids = np.sort(
			np.unique([x[2] for x in audiopaths_and_text])
		)
		d = {int(speaker_ids[i]): i for i in range(len(speaker_ids))}
		print("Number of speakers :", len(d))
		return d


	def get_mel(self, audio):
		audio_norm = audio / self.max_wav_value
		audio_norm = tf.expand_dims(audio_norm, axis=0)
		melspec = self.stft.mel_spectrogram(audio_norm)
		return melspec


	def get_speaker_id(self, speaker_id):
		# return torch.LongTensor([self.speaker_ids[int(speaker_id)]])
		return tf.convert_to_tensor(
			[self.speaker_ids[int(speaker_id)]], dtype=tf.int64
		)


	def get_text(self, text):
		text = _clean_text(text, self.text_cleaners)
		words = re.findall(r'\S*\{.*?\}\S*|\S+', text)
		text = ' '.join([get_arpabet(word, self.cmudict)
						 if random.random() < self.p_arpabet else word
						 for word in words])
		text_norm = tf.convert_to_tensor(
			text_to_sequence(text), dtype=tf.int64
		)
		return text_norm


	def generator(self):
		self.max_input_len = 0
		self.max_target_len = 0

		for idx in tqdm(range(len(self.audiopaths_and_text))):
			mels, speaker_id, text_encoded, attn_prior = self.__getitem__(idx)
			self.max_input_len = max(
				tf.shape(text_encoded)[0], self.max_input_len
			)
			self.max_target_len = max(
				tf.shape(mels)[0], self.max_target_len
			)
			yield mels, speaker_id, text_encoded, attn_prior


	def __getitem__(self, index):
		# Read audio and text
		audiopath, text, speaker_id = self.audiopaths_and_text[index]
		# audio, sampling_rate = load_wav_to_torch(audiopath)
		audio, sampling_rate = load_wav_to_tensorflow(audiopath)
		if sampling_rate != self.sampling_rate:
			raise ValueError("{} SR doesn't match target {} SR".format(
				sampling_rate, self.sampling_rate))

		mel = self.get_mel(audio)
		text_encoded = self.get_text(text)
		speaker_id = self.get_speaker_id(speaker_id)
		attn_prior = None
		if self.use_attn_prior:
			attn_prior = self.compute_attention_prior(
				# audiopath, mel.shape[1], text_encoded.shape[0])
				audiopath, mel.shape[0], text_encoded.shape[0])

		return (mel, speaker_id, text_encoded, attn_prior)


	def __len__(self):
		return len(self.audiopaths_and_text)


class DataCollate:
	def __init__(self, n_frames_per_step=1, use_attn_prior=False):
		self.n_frames_per_step = n_frames_per_step
		self.use_attn_prior = use_attn_prior
		self.max_input_len = 0
		self.max_target_len = 0


	def __call__(self, mel, speaker_id, text_encoded, attn_prior):
		# The following assertions make sure that the text and mel
		# spectrogram lengths do not exceed the maximums set in the
		# object.
		assert self.max_input_len >= text_encoded.shape[0], f"Input encoded text length exceeds maximum length ({self.max_input_len}): Received {text_encoded.shape[0]}"
		assert self.max_target_len >= mel.shape[0], f"Target mel specrogram length exceeds maximum length ({self.max_target_len}): Received {mel.shape[0]}"

		# with tf.device("/cpu:0"):
		# Right zero-pad all one-hot text sequences to max input
		# length.
		text_padded = np.zeros((self.max_input_len,), dtype=np.int_)
		text_padded[:text_encoded.shape[0]] = text_encoded
		text_padded = tf.convert_to_tensor(text_padded, dtype=tf.int64)
		
		# Right zero-pad mel-spec.
		n_mel_channels = mel.shape[1]
		if self.max_target_len % self.n_frames_per_step != 0:
			max_target_len = self.max_target_len
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0

		# Include mel padded, gate padded, and speaker ids.
		mel_padded = np.zeros(
			(self.max_target_len, n_mel_channels), dtype=np.float32
		)
		gate_padded = np.zeros((self.max_target_len), dtype=np.float32)

		# NOTE: use_attn_prior = None, self.use_attn_prior is not
		# handled in the original data.py in the Nvidia Flowtron repo.
		attn_prior_padded = None
		if self.use_attn_prior:
			attn_prior_padded = np.zeros(
				(self.max_target_len, self.max_input_len)
			)
			attn_prior_padded[
				:attn_prior.shape[0], :attn_prior.shape[1]
			] = attn_prior
			attn_prior_padded = tf.convert_to_tensor(
				attn_prior_padded, dtype=tf.float32
			)
		mel_padded[:mel.shape[0], :] = mel
		mel_padded = tf.convert_to_tensor(mel_padded)
		gate_padded[mel.shape[0] - 1:] = 1
		gate_padded = tf.convert_to_tensor(gate_padded, dtype=tf.int64)

		# Convert remaining values to tensors.
		speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.int64)
		input_lengths = tf.convert_to_tensor(
			text_encoded.shape[0], dtype=tf.int64
		)
		output_lengths = tf.convert_to_tensor(
			mel.shape[0], dtype=tf.int64
		)

		# Outputs are mel padded, speaker ids, text padded, input
		# lengths, output lengths, gate padded, attn prior padded.
		# Output dtypes are tf.float32, tf.int64, tf.int64, tf.int64,
		# tf.int64, tf.float32, tf.float32.
		# Output shapes are (None, n_mel_channels), (1,), (None,),
		# (1,), (1,), (None,), (None, None) 
		return (mel_padded, speaker_id, text_padded, input_lengths,
				output_lengths, gate_padded, attn_prior_padded)


	def update_max_len(self, max_input_len, max_target_len):
		# Update the maximum input ((encoded) text) and target/output
		# (mel-spectrogram) lengths.
		self.max_input_len = max_input_len
		self.max_target_len = max_target_len


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-f', '--filelist', type=str,
						help='List of files to generate mels')
	parser.add_argument('-o', '--output_dir', type=str,
						help='Output directory')
	args = parser.parse_args()

	with open(args.config) as f:
		data = f.read()
	data_config = json.loads(data)["data_config"]
	data_config["filelist_path"] = args.filelist
	mel2samp = Data(**data_config)

	# Make directory if it doesn't exist
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
		os.chmod(args.output_dir, 0o775)

	filepaths_and_text = load_filepaths_and_text(args.filelist)
	for (filepath, text, speaker_id) in filepaths_and_text:
		print("speaker id", speaker_id)
		print("text", text)
		print("text encoded", mel2samp.get_text(text))
		# audio, sr = load_wav_to_torch(filepath)
		audio, sr = load_wav_to_tensorflow(filepath)
		melspectrogram = mel2samp.get_mel(audio)
		filename = os.path.basename(filepath)
		# new_filepath = args.output_dir + '/' + filename + '.pt'
		new_filepath = args.output_dir + '/' + filename + '.npy'
		print(new_filepath)
		# torch.save(melspectrogram, new_filepath)
		np.save(new_filepath, melspectrogram.numpy())
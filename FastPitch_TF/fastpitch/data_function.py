# data_function.py
# Classes and functions to load and preprocess dataset.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import re
import os
import argparse
import json
import random
import functools
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import betabinom
from common.text.text_processing import TextProcessing
from common.audio_processing_tf import STFT
# from common.utils import load_wav_to_tensorflow
# from common.utils import load_filepaths_and_text


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


class BetaBinomialInterpolator:
	"""
	Interpolates alignment prior matrices to save computation.

	Calculating beta-binomial priors is costly. Instead cache popular
	sizes and use img interpolation to get priors faster.
	"""
	def __init__(self, round_mel_len_to=100, round_text_len_to=20):
		self.round_mel_len_to = round_mel_len_to
		self.round_text_len_to = round_text_len_to
		self.bank = functools.lru_cache(
			beta_binomial_prior_distribution
		)


	def round(self, val, to):
		return max(1, int(np.round((val + 1) / to))) * to


	def __call__(self, w, h):
		bw = self.round(w, to=self.round_mel_len_to)
		bh = self.round(h, to=self.round_text_len_to)
		ret = ndimage.zoom(
			# self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1
			tf.transpose(self.bank(bw, bh)).numpy(), 
			zoom=(w / bw, h / bh), order=1
		)
		assert ret.shape[0] == w, ret.shape
		assert ret.shape[1] == h, ret.shape
		return tf.convert_to_tensor(ret, dtype=tf.float32)


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


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
			normalize_std=None, n_formants=1):
	if type(normalize_mean) is float or type(normalize_mean) is list:
		normalize_mean = torch.tensor(normalize_mean)
	if type(normalize_std) is float or type(normalize_std) is list:
		normalize_std = torch.tensor(normalize_std)

	if method == 'pyin':
		# snd, sr = librosa.load(wav)
		snd, sr = load_wav_to_tensorflow(wav)
		pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
			snd.numpy(), fmin=librosa.note_to_hz('C2'),
			fmax=librosa.note_to_hz('C7'), frame_length=1024)
		assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

		pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
		# pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
		# pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))
		pitch_mel = tf.expand_dims(tf.convert_to_tensor(pitch_mel), 0)
		pad = [0, mel_len - tf.shape(pitch_mel)[1]]
		pitch_mel = tf.pad(pitch_mel, pad)

		if n_formants > 1:
			raise NotImplementedError
	else:
		raise ValueError

	# pitch_mel = pitch_mel.float()
	pitch_mel = tf.cast(pitch_mel, dtype=tf.float32)

	if normalize_mean is not None:
		assert normalize_std is not None
		pitch_mel = normalize_pitch(
			pitch_mel, normalize_mean, normalize_std
		)

	return pitch_mel


def normalize_pitch(pitch, mean, std):
	zeros = (pitch == 0.0)
	pitch -= mean[:, None]
	pitch /= std[:, None]
	pitch[zeros] = 0.0
	return pitch


class Data:
	def __init__(self, filelist_path, text_cleaners, n_mel_channels,
			symbol_set="english_basic", p_arpabet=1.0, n_speakers=1,
			load_mel_from_disk=True, load_pitch_from_disk=True,
			pitch_mean=214.72203, pitch_std=65.72038, 
			max_wav_value=None, sampling_rate=None, filter_length=None,
			hop_length=None, win_length=None, mel_fmin=None,
			mel_fmax=None, prepend_space_to_text=False,
			append_space_to_text=False, pitch_online_dir=None,
			betabinomial_online_dir=None, 
			use_betabinomial_interpolator=True,
			pitch_online_method="pyin", **ignored):
		# Expect a list of filenames.
		if type(filelist_path) is str:
			filelist_path = [filelist_path]

		self.audiopaths_and_text = load_filepaths_and_text(
			filelist_path
		)
		self.load_mel_from_disk = load_mel_from_disk
		if not load_mel_from_disk:
			self.max_wav_value = max_wav_value
			self.sampling_rate = sampling_rate
			self.stft = STFT(
				filter_length=filter_length, frame_step=hop_length,
				frame_length=win_length, sampling_rate=sampling_rate,
				mel_fmin=mel_fmin, mel_fmax=mel_fmax
			)

		self.load_pitch_from_disk = load_pitch_from_disk

		self.prepend_space_to_text = prepend_space_to_text
		self.append_space_to_text = append_space_to_text

		assert p_arpabet == 0.0 or p_arpabet == 1.0, (
			"Only 0.0 and 1.0 p_arpabet is currently supported. "
			"Variable probability breaks caching of betabinomial "
			"matrices."
		)

		self.tp = TextProcessing(
			symbol_set, text_cleaners, p_arpabet=p_arpabet
		)
		self.n_speakers = n_speakers
		self.pitch_tmp_dir = pitch_online_dir
		self.f0_method = pitch_online_method
		self.betabinomial_tmp_dir = betabinomial_online_dir
		self.use_betabinomial_interpolator = use_betabinomial_interpolator

		if use_betabinomial_interpolator:
			self.betabinomial_interpolator = BetaBinomialInterpolator()

		expected_columns = (
			2 + int(load_pitch_from_disk) + (n_speakers > 1)
		)

		assert not (
			load_pitch_from_disk and self.pitch_tmp_dir is not None
		)

		if len(self.audiopaths_and_text[0]) < expected_columns:
			raise ValueError(
				f"Expected {expected_columns} columns in audiopaths file. "
				"The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]"
			)

		if len(self.audiopaths_and_text[0]) > expected_columns:
			print("WARNING: Audiopaths file has more columns than expected")

		to_tensor = lambda x: tf.convert_to_tensor([x]) if type(x) is float else x
		self.pitch_mean = to_tensor(pitch_mean)
		self.pitch_std = to_tensor(pitch_std)


	def __getitem__(self, index):
		# Separate filename and text
		if self.n_speakers > 1:
			audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
			speaker = int(speaker)
		else:
			audiopath, *extra, text = self.audiopaths_and_text[index]
			speaker = None

		mel = self.get_mel(audiopath)
		text = self.get_text(text)
		pitch = self.get_pitch(index, tf.shape(mel)[0])
		energy = tf.norm(tf.cast(mel, dtype=tf.float32), axis=1, ord=0)
		attn_prior = self.get_prior(
			index, tf.shape(mel)[0], tf.shape(text)[0]
		)

		assert tf.shape(pitch)[0] == tf.shape(mel)[0]

		# No higher formants?
		if len(tf.shape(pitch)) == 1:
			pitch = pitch[None, :]

		return (
			text, mel, len(text), pitch, energy, speaker, attn_prior,
			audiopath
		)


	def __len__(self):
		return len(self.audiopaths_and_text)


	def get_mel(self, filename):
		pass


	def get_text(self, text):
		text = self.tp.encode_text(text)
		space = [self.tp.encode_text("A A")[1]]
		
		if self.prepend_space_to_text:
			text = space + text

		if self.append_space_to_text:
			text = text + space

		return tf.convert_to_tensor(text, dtype=tf.int64)


	def get_prior(self, index, mel_len, text_len):
		if self.use_betabinomial_interpolator:
			return tf.convert_to_tensor(
				self.betabinomial_interpolator(mel_len, text_len)
			)

		if self.betabinomial_tmp_dir is not None:
			audiopath, *_ = self.audiopaths_and_text[index]
			# fname = Path(audiopath).relative_to(self.)
			pass

		attn_prior = beta_binomial_prior_distribution(
			text_len, mel_len
		)

		if self.betabinomial_tmp_dir is not None:
			pass

		return attn_prior


	def get_pitch(self, index, mel_len=None):
		audiopath, *fields = self.audiopaths_and_text[index]

		if self.n_speakers > 1:
			spk = int(fields[-1])
		else:
			spk = 0

		if self.load_pitch_from_disk:
			pitchpath = fields[0]
			pitch = tf.convert_to_tensor(np.load(pitchpath))
			if self.pitch_mean is not None:
				assert self.pitch_std is not None
				pitch = normalize_pitch(
					pitch, pitch_mean, self.pitch_std
				)
			return pitch

		if self.pitch_tmp_dir is not None:
			pass

		# No luck so far - calculate.
		wav = audiopath
		if not wav.endswith(".wav"):
			wav = re.sub("/mels/", "/wavs/", wav)
			wav = re.sub(".npy", ".wav", wav)

		pitch_mel = estimate_pitch(
			wav, mel_len, self.f0_method, self.pitch_mean,
			self.pitch_std
		)

		if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
			cached_fpath.parent.mkdir(parents=True, exist_ok=True)
			np.save(pitch_mel, cached_fpath)
		
		return pitch_mel


	def collate_fn(self, batch):
		pass



class Data_Archived:
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

		# Check for preprocessed data.
		all_exist = True
		dataset_path = f"./processed_dataset"
		if os.path.exists(dataset_path):
			# Validate all files for each sample exist.
			print("Indexing processed dataset files...")
			for idx in range(len(self.audiopaths_and_text)):
				basename = os.path.basename(
					self.audiopaths_and_text[idx][0]
				).rstrip(".wav")
				mel_path = dataset_path + "/" + basename + "_mel.npy"
				speaker_id_path = dataset_path + "/" + basename +\
					"_speaker_id.npy"
				text_encoded_path = dataset_path + "/" + basename +\
					"_text_encoded.npy"
				input_len_path = dataset_path + "/" + basename +\
					"_input_len.npy"
				output_len_path = dataset_path + "/" + basename +\
					"_output_len.npy"
				gate_path = dataset_path + "/" + basename + "_gate.npy"
				attn_prior_path = dataset_path + "/" + basename +\
					"_attn_prior.npy"
				all_exist = all([
					os.path.exists(mel_path), 
					os.path.exists(text_encoded_path),
					os.path.exists(input_len_path),
					os.path.exists(output_len_path),
					os.path.exists(gate_path),
					os.path.exists(attn_prior_path)
				])

				# Break the loop if any of the files are missing.
				if not all_exist:
					break
		else:
			all_exist = False

		# Preprocess the dataset only if there are missing files for
		# the dataset.
		if all_exist:
			# Load each numpy array from the respective files and
			# convert it to tensorflow tensors.
			print("Loading pre-computed data...")
			for idx in tqdm(range(len(self.audiopaths_and_text))):
				basename = os.path.basename(
					self.audiopaths_and_text[idx][0]
				).rstrip(".wav")
				mel_path = dataset_path + "/" + basename + "_mel.npy"
				speaker_id_path = dataset_path + "/" + basename +\
					"_speaker_id.npy"
				text_encoded_path = dataset_path + "/" + basename +\
					"_text_encoded.npy"
				input_len_path = dataset_path + "/" + basename +\
					"_input_len.npy"
				output_len_path = dataset_path + "/" + basename +\
					"_output_len.npy"
				gate_path = dataset_path + "/" + basename + "_gate.npy"
				attn_prior_path = dataset_path + "/" + basename +\
					"_attn_prior.npy"

				mels = tf.convert_to_tensor(
					np.load(mel_path), dtype=tf.float32
				)
				speaker_id = tf.convert_to_tensor(
					np.load(speaker_id_path), dtype=tf.int64
				)
				text_encoded = tf.convert_to_tensor(
					np.load(text_encoded_path), dtype=tf.int64	
				)
				input_lengths = tf.convert_to_tensor(
					np.load(input_len_path), dtype=tf.int64
				)
				output_lengths = tf.convert_to_tensor(
					np.load(output_len_path), dtype=tf.int64
				)
				gate = tf.convert_to_tensor(
					np.load(gate_path), dtype=tf.int64
				)
				attn_prior = tf.convert_to_tensor(
					np.load(attn_prior_path), dtype=tf.float32
				)
				yield (
					mels, speaker_id, text_encoded, input_lengths, 
					output_lengths, gate, attn_prior
				)
		else:
			# Compute the maximum input (text) and target
			# (mel-spectrogram) lengths.
			print("Isolating max input and target lengths...")
			for idx in tqdm(range(len(self.audiopaths_and_text))):
				mels, _, text_encoded, _ = self.__getitem__(idx)
				self.max_input_len = max(
					tf.shape(text_encoded)[0], self.max_input_len
				)
				self.max_target_len = max(
					tf.shape(mels)[0], self.max_target_len
				)

			# Update the collate function max lengths. Must have the 
			# collate function initialized before.
			assert hasattr(self, "collate_fn"), "Collate function for dataset not set."
			self.collate_fn.update_max_len(
				self.max_input_len, self.max_target_len
			)

			# Apply data collate function to each item in the dataset.
			print("Applying data collator function...")
			for idx in tqdm(range(len(self.audiopaths_and_text))):
				mels, speaker_id, text_encoded, attn_prior = self.__getitem__(idx)
				yield self.collate_fn(
					mels, speaker_id, text_encoded, attn_prior
				)


	def set_collate_fn(self, collate_fn):
		self.collate_fn = collate_fn


	def save_processed_dataset(self, overide=False):
		# Verify that the collate function is set. The generator
		# function wont work if it is not set.
		assert self.collate_fn is not None, "Collate function for dataset is not set."

		# Main folder to save the data for the dataset.
		all_exist = True
		dataset_path = f"./processed_dataset"
		os.makedirs(dataset_path, exist_ok=True)
		if os.path.exists(dataset_path):
			# Validate all files for each sample exist.
			print("Indexing processed dataset files...")
			for idx in range(len(self.audiopaths_and_text)):
				basename = os.path.basename(
					self.audiopaths_and_text[idx][0]
				).rstrip(".wav")
				mel_path = dataset_path + "/" + basename + "_mel.npy"
				speaker_id_path = dataset_path + "/" + basename +\
					"_speaker_id.npy"
				text_encoded_path = dataset_path + "/" + basename +\
					"_text_encoded.npy"
				input_len_path = dataset_path + "/" + basename +\
					"_input_len.npy"
				output_len_path = dataset_path + "/" + basename +\
					"_output_len.npy"
				gate_path = dataset_path + "/" + basename + "_gate.npy"
				attn_prior_path = dataset_path + "/" + basename +\
					"_attn_prior.npy"
				all_exist = all([
					os.path.exists(mel_path), 
					os.path.exists(text_encoded_path),
					os.path.exists(input_len_path),
					os.path.exists(output_len_path),
					os.path.exists(gate_path),
					os.path.exists(attn_prior_path)
				])

				# Break the loop if any of the files are missing.
				if not all_exist:
					break

		# Exit the function early if the data has already been saved
		# and there is no intention on overriding existing saved data.
		# This should help with performance on reducing redundant 
		# operations.
		if all_exist and not overide:
			return

		# Iterate through each element yielded by the generator. Keep
		# track of the index (index value maps 1:1 with dataset
		# audiopaths and text since the shuffling happens before this
		# step and the path names are irrelevant after).
		idx = 0
		for tensor_tuple in self.generator():
			# Isolate the base name of the file associated to the data.
			# Use that base name to name the respective npy files for
			# each data array.
			basename = os.path.basename(
				self.audiopaths_and_text[idx][0]
			).rstrip(".wav")
			mel_path = dataset_path + "/" + basename + "_mel.npy"
			speaker_id_path = dataset_path + "/" + basename +\
				"_speaker_id.npy"
			text_encoded_path = dataset_path + "/" + basename +\
				"_text_encoded.npy"
			input_len_path = dataset_path + "/" + basename +\
				"_input_len.npy"
			output_len_path = dataset_path + "/" + basename +\
				"_output_len.npy"
			gate_path = dataset_path + "/" + basename + "_gate.npy"
			attn_prior_path = dataset_path + "/" + basename +\
				"_attn_prior.npy"

			# Unpack tuple of tensors from generator.
			(
				mels, speaker_id, text_encoded, input_len, output_len, 
				gate, attn_prior
			) = tensor_tuple
			np.save(mel_path, mels.numpy())
			np.save(speaker_id_path, speaker_id.numpy())
			np.save(text_encoded_path, text_encoded.numpy())
			np.save(input_len_path, input_len.numpy())
			np.save(output_len_path, output_len.numpy())
			np.save(gate_path, gate.numpy())
			np.save(attn_prior_path, attn_prior.numpy())

			# Increment index.
			idx += 1


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
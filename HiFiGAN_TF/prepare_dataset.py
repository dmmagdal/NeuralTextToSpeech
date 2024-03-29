# prepare_dataset.py


import json
import time
from pathlib import Path
import tensorflow as tf
import tqdm
from data import Data, DataOld
from hparams import HParams


def main():
	# text_cleaners = ['english_cleaners_v2'] # Not needed.
	dataset_path = './ljspeech_train'
	filelist = './filelists/ljs_audio_text_train_v3.txt'
	# dataset_path = './ljspeech_valid'
	# filelist = './filelists/ljs_audio_text_val.txt'

	# Load parameters from config.json.
	config = 'config_v3.json'
	with open(config, 'r') as f:
		hparams = json.load(f)
	hparams = HParams(**hparams)

	extract_mels = True
	max_wav_value = 32768.0

	# mel extraction
	# n_speakers = 1
	# max_wav_value = 32768.0
	# sampling_rate = 22050
	# filter_length = 1024
	# hop_length = 256
	# win_length = 1024
	# mel_fmin = 0.0
	# mel_fmax = 8000.0
	# n_mel_channels = 80
	# f0_method = 'pyin'
	# batch_size = 1

	dataset_old = DataOld(
		dataset_path, # dataset_path
		filelist, # filelist_path
		# params.cmudict_path, # cmudict_path
		# n_mel_channels=n_mel_channels,
		n_mel_channels=hparams.num_mels,
		n_speakers=1, # Manually set because config.json does not have variable (& it isnt necessary for the vocoder)
		load_mel_from_disk=False,
		# sampling_rate=sampling_rate,
		sampling_rate=hparams.sampling_rate,
		# filter_length=filter_length,
		filter_length=hparams.n_fft,
		hop_length=hparams.hop_size,
		win_length=hparams.win_size,
		# mel_fmin=mel_fmin,
		# mel_fmax=mel_fmax
		mel_fmin=hparams.fmin,
		mel_fmax=hparams.fmax
	)
	dataset = Data(
		filelist, # filelist_path
		segment_size=hparams.segment_size,
		# n_mel_channels=n_mel_channels,
		n_mel_channels=hparams.num_mels,
		# filter_length=filter_length,
		filter_length=hparams.n_fft,
		hop_length=hparams.hop_size,
		win_length=hparams.win_size,
		# sampling_rate=sampling_rate,
		sampling_rate=hparams.sampling_rate,
		# mel_fmin=mel_fmin,
		# mel_fmax=mel_fmax
		mel_fmin=hparams.fmin,
		mel_fmax=hparams.fmax,
		n_speakers=1, # Manually set because config.json does not have variable (& it isnt necessary for the vocoder)
	)

	# Additional code.
	print(dataset_old.__getitem__(0))
	# for i in dataset_old.__getitem__(0):
	# 	print(i)
	# for i in range(dataset_old.__len__()):
	# 	dataset_old.__getitem__(i)
	print(dataset.__getitem__(0))
	# for i in dataset.__getitem__(0):
	# 	print(i)
	# for i in range(dataset.__len__()):
	# 	dataset.__getitem__(i)

	# Outputs are the following:
	# -> padded encoded texts (dtype=tf.int64, 
	#	shape=(max_input_length))
	# -> input lengths (dtype=tf.int64, shape=())
	# -> padded mel spectrograms (dtype=tf.float32,
	#	shape=(max_target_length, n_mel_channels))
	# -> output lengths (dtype=tf.int64, shape=())
	# -> text length (dtype=tf.int64, shape=())
	# -> padded pitch (dtype=tf.float32, 
	#	shape=(n_formants, max_target_lengths + 4))
	# -> padded energy (dtype=tf.float32,
	#	shape=(max_target_length,))
	# -> speaker id (dtype=tf.int64, shape=())
	# -> padded attention priors (dtype=tf.float32,
	#	shape=(max_target_length, max_input_length))
	# -> audiopath (dtype=tf.string, shape=())

	# data = tf.data.Dataset.from_generator( # Use in eager execution.
	# 	dataset.generator,
	# 	args=(),
	# 	output_signature=(
	# 		tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
	# 		tf.TensorSpec(
	# 			# shape=(None, n_mel_channels), dtype=tf.float32
	# 			shape=(None, params.n_feats), dtype=tf.float32
	# 		),														# mel
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
	# 	)
	# )
	# data_old = tf.data.Dataset.from_generator( # Use in eager execution.
	# 	dataset_old.generator,
	# 	args=(),
	# 	output_signature=(
	# 		tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),							# output_lengths
	# 	)
	# )
	data = tf.data.Dataset.from_generator( # Use in eager execution.
		dataset.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel
			tf.TensorSpec(shape=(None,), dtype=tf.float32),						# audio
			tf.TensorSpec(shape=(), dtype=tf.string),							# filename
			tf.TensorSpec(shape=(None, hparams.num_mels), dtype=tf.float32),	# mel_loss
		)
	)
	# print(list(data_old.as_numpy_iterator())[0])
	print(list(data.as_numpy_iterator())[0])
	x = list(data.as_numpy_iterator())[:3]
	for i in range(len(x)):
		y = x[i]
		print(f"{tf.shape(y[0])}, {tf.shape(y[1])}, {tf.shape(y[2])}, {tf.shape(y[3])}")
	# print(json.dumps(list(data.as_numpy_iterator())[:3]))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
# prepare_dataset.py


import time
from pathlib import Path
import tensorflow as tf
import tqdm
# from fastpitch.data_function import TTSCollater, TTSDataset
from data_function import Data


def main():
	text_cleaners = ['english_cleaners_v2']
	dataset_path = './ljspeech_train'
	# filelist = './filelists/ljs_audio_text_train_v3.txt'
	filelist = './filelists/ljs_audio_text_val.txt'

	extract_mels = True
	extract_pitch = True
	save_alignment_priors = True

	# mel extraction
	n_speakers = 1
	max_wav_value = 32768.0
	sampling_rate = 22050
	filter_length = 1024
	hop_length = 256
	win_length = 1024
	mel_fmin = 0.0
	mel_fmax = 8000.0
	n_mel_channels = 80
	f0_method = 'pyin'
	batch_size = 1

	# dataset = TTSDataset(
	dataset = Data(
		dataset_path, 
		filelist, 
		text_cleaners=text_cleaners,
		n_mel_channels=n_mel_channels,
		p_arpabet=0.0,
		n_speakers=n_speakers,
		load_mel_from_disk=False,
		load_pitch_from_disk=False,
		pitch_mean=None,
		pitch_std=None,
		max_wav_value=max_wav_value,
		sampling_rate=sampling_rate,
		filter_length=filter_length,
		hop_length=hop_length,
		win_length=win_length,
		mel_fmin=mel_fmin,
		mel_fmax=mel_fmax,
		betabinomial_online_dir=None,
		pitch_online_dir=None,
		pitch_online_method=f0_method
	)

	# Additional code.
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

	data = tf.data.Dataset.from_generator(
		dataset.generator,
		args=(),
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
			tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			tf.TensorSpec(
				shape=(None, n_mel_channels), dtype=tf.float32
			),														# mel
			tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			tf.TensorSpec(shape=(), dtype=tf.int64),				# text_length
			tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# pitch
			tf.TensorSpec(shape=(None,), dtype=tf.float32),			# energy
			tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
			tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# attn prior
			tf.TensorSpec(shape=(), dtype=tf.string),				# audiopath
		)
	)
	print(list(data.as_numpy_iterator())[0])

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
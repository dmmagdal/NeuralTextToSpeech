# prepare_dataset.py


import time
from pathlib import Path
import tensorflow as tf
import tqdm
from data import Data
from params import params


def main():
	dataset_path = './ljspeech_train'
	filelist = './filelists/ljs_audio_text_train_v3.txt'
	# filelist = './filelists/ljs_audio_text_val.txt'

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

	dataset = Data(
		dataset_path, # dataset_path
		filelist, # filelist_path
		params,
		# n_mel_channels=n_mel_channels,
		n_mel_channels=params.n_mels,
		# n_speakers=params.n_spks,
		n_speakers=1, # Manually added because this tests LJSpeech dataset
		load_mel_from_disk=False,
		# sampling_rate=sampling_rate,
		sampling_rate=params.sample_rate,
		# filter_length=filter_length,
		filter_length=params.n_fft,
		hop_length=params.hop_length,
		win_length=params.win_length,
		# mel_fmin=mel_fmin,
		# mel_fmax=mel_fmax
		mel_fmin=params.f_min,
		mel_fmax=params.f_max, 
		from_gtzan=False
	)

	# Additional code.
	print(dataset.__getitem__(0))
	# for i in dataset.__getitem__(0):
	# 	print(i)
	# for i in range(dataset.__len__()):
	# 	dataset.__getitem__(i)

	# Outputs are the following:
	# -> padded audio (dtype=tf.float32, 
	#	shape=(max_input_length))
	# -> padded mel spectrograms (dtype=tf.float32,
	#	shape=(max_target_length, n_mel_channels))

	# Output signature.
	signature = (
		tf.TensorSpec(shape=(None,), dtype=tf.float32),		# audio
		tf.TensorSpec(
			# shape=(None, n_mel_channels), dtype=tf.float32
			shape=(None, params.n_mels), dtype=tf.float32
		),													# mel
	)
	if params.unconditional:
		# Output signature is different if using unconditional model.
		signature = (
			tf.TensorSpec(shape=(None,), dtype=tf.float32),	# audio
		)

	data = tf.data.Dataset.from_generator( # Use in eager execution.
		dataset.generator,
		args=(),
		output_signature=signature
	)
	print(list(data.as_numpy_iterator())[0])

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
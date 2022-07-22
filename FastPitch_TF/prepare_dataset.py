# prepare_dataset.py


import time
from pathlib import Path
import tensorflow as tf
import tqdm
from fastpitch.data_function import TTSCollater, TTSDataset


def main():
	text_cleaners = ['english_cleaners_v2']
	dataset_path = 'some/path'
	filelist = 'list'

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

	dataset = TTSDataset(
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

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
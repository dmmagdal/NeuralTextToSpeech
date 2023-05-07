# train.py


import os
import argparse
import tensorflow as tf



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--group_name', default=None)
	parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
	parser.add_argument('--input_mels_dir', default='ft_dataset')
	# parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')     # Not needed/used
	# parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt') # Not needed/used
	parser.add_argument('--checkpoint_path', default='cp_hifigan')
	parser.add_argument('--config', default='')
	parser.add_argument('--training_epochs', default=3100, type=int)
	parser.add_argument('--stdout_interval', default=5, type=int)
	parser.add_argument('--checkpoint_interval', default=5000, type=int)
	parser.add_argument('--summary_interval', default=100, type=int)
	parser.add_argument('--validation_interval', default=1000, type=int)
	parser.add_argument('--fine_tuning', default=False, type=bool)


if __name__ == '__main__':
	main()
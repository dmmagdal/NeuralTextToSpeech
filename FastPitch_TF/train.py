# train.py


import os
import time
import copy
import json
import argparse
from collections import defaultdict, OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import models
# from fastpitch.attn_loss_function import AttentionBinarizationLoss
from attn_loss_function import AttentionBinarizationLoss
# from fastpitch.data_function import Data
from data_function import Data
# from fastpitch.loss_function import FastPitchLoss
from loss_function import FastpitchLoss
from model import FastPitch
from models import get_fastpitch_config, parse_model_args

tf.debugging.experimental.enable_dump_debug_info(
	"./tmp/tfdbg2_logdir",
	tensor_debug_mode="FULL_HEALTH",
	circular_buffer_size=-1
)


def parse_args(parser):
	parser.add_argument('-o', '--output', type=str, required=True,
						help='Directory to save checkpoints')
	parser.add_argument('-d', '--dataset-path', type=str, default='./',
						help='Path to dataset')
	parser.add_argument('--log-file', type=str, default=None,
						help='Path to a DLLogger log file')

	train = parser.add_argument_group('training setup')
	train.add_argument('--epochs', type=int, required=True,
					   help='Number of total epochs to run')
	train.add_argument('--epochs-per-checkpoint', type=int, default=50,
					   help='Number of epochs per checkpoint')
	train.add_argument('--checkpoint-path', type=str, default=None,
					   help='Checkpoint path to resume training')
	train.add_argument('--keep-milestones', default=list(range(100, 1000, 100)),
					   type=int, nargs='+',
					   help='Milestone checkpoints to keep from removing')
	train.add_argument('--resume', action='store_true',
					   help='Resume training from the last checkpoint')
	train.add_argument('--seed', type=int, default=1234,
					   help='Seed for PyTorch random number generators')
	train.add_argument('--amp', action='store_true',
					   help='Enable AMP')
	train.add_argument('--cuda', action='store_true',
					   help='Run on GPU using CUDA')
	train.add_argument('--cudnn-benchmark', action='store_true',
					   help='Enable cudnn benchmark mode')
	train.add_argument('--ema-decay', type=float, default=0,
					   help='Discounting factor for training weights EMA')
	train.add_argument('--grad-accumulation', type=int, default=1,
					   help='Training steps to accumulate gradients for')
	train.add_argument('--kl-loss-start-epoch', type=int, default=250,
					   help='Start adding the hard attention loss term')
	train.add_argument('--kl-loss-warmup-epochs', type=int, default=100,
					   help='Gradually increase the hard attention loss term')
	train.add_argument('--kl-loss-weight', type=float, default=1.0,
					   help='Gradually increase the hard attention loss term')
	train.add_argument('--benchmark-epochs-num', type=int, default=20,
						help='Number of epochs for calculating final stats')

	opt = parser.add_argument_group('optimization setup')
	opt.add_argument('--optimizer', type=str, default='lamb',
					 help='Optimization algorithm')
	opt.add_argument('-lr', '--learning-rate', type=float, required=True,
					 help='Learing rate')
	opt.add_argument('--weight-decay', default=1e-6, type=float,
					 help='Weight decay')
	opt.add_argument('--grad-clip-thresh', default=1000.0, type=float,
					 help='Clip threshold for gradients')
	opt.add_argument('-bs', '--batch-size', type=int, required=True,
					 help='Batch size per GPU')
	opt.add_argument('--warmup-steps', type=int, default=1000,
					 help='Number of steps for lr warmup')
	opt.add_argument('--dur-predictor-loss-scale', type=float,
					 default=1.0, help='Rescale duration predictor loss')
	opt.add_argument('--pitch-predictor-loss-scale', type=float,
					 default=1.0, help='Rescale pitch predictor loss')
	opt.add_argument('--attn-loss-scale', type=float,
					 default=1.0, help='Rescale alignment loss')

	data = parser.add_argument_group('dataset parameters')
	data.add_argument('--training-files', type=str, nargs='*', required=True,
					  help='Paths to training filelists.')
	data.add_argument('--validation-files', type=str, nargs='*',
					  required=True, help='Paths to validation filelists')
	data.add_argument('--text-cleaners', nargs='*',
					  default=['english_cleaners'], type=str,
					  help='Type of text cleaners for input text')
	data.add_argument('--symbol-set', type=str, default='english_basic',
					  help='Define symbol set for input text')
	data.add_argument('--p-arpabet', type=float, default=0.0,
					  help='Probability of using arpabets instead of graphemes '
						   'for each word; set 0 for pure grapheme training')
	data.add_argument('--heteronyms-path', type=str, default='cmudict/heteronyms',
					  help='Path to the list of heteronyms')
	data.add_argument('--cmudict-path', type=str, default='cmudict/cmudict-0.7b',
					  help='Path to the pronouncing dictionary')
	data.add_argument('--prepend-space-to-text', action='store_true',
					  help='Capture leading silence with a space token')
	data.add_argument('--append-space-to-text', action='store_true',
					  help='Capture trailing silence with a space token')

	cond = parser.add_argument_group('data for conditioning')
	cond.add_argument('--n-speakers', type=int, default=1,
					  help='Number of speakers in the dataset. '
						   'n_speakers > 1 enables speaker embeddings')
	cond.add_argument('--load-pitch-from-disk', action='store_true',
					  help='Use pitch cached on disk with prepare_dataset.py')
	cond.add_argument('--pitch-online-method', default='pyin',
					  choices=['pyin'],
					  help='Calculate pitch on the fly during trainig')
	cond.add_argument('--pitch-online-dir', type=str, default=None,
					  help='A directory for storing pitch calculated on-line')
	cond.add_argument('--pitch-mean', type=float, default=214.72203,
					  help='Normalization value for pitch')
	cond.add_argument('--pitch-std', type=float, default=65.72038,
					  help='Normalization value for pitch')
	cond.add_argument('--load-mel-from-disk', action='store_true',
					  help='Use mel-spectrograms cache on the disk')  # XXX

	audio = parser.add_argument_group('audio parameters')
	audio.add_argument('--max-wav-value', default=32768.0, type=float,
					   help='Maximum audiowave value')
	audio.add_argument('--sampling-rate', default=22050, type=int,
					   help='Sampling rate')
	audio.add_argument('--filter-length', default=1024, type=int,
					   help='Filter length')
	audio.add_argument('--hop-length', default=256, type=int,
					   help='Hop (stride) length')
	audio.add_argument('--win-length', default=1024, type=int,
					   help='Window length')
	audio.add_argument('--mel-fmin', default=0.0, type=float,
					   help='Minimum mel frequency')
	audio.add_argument('--mel-fmax', default=8000.0, type=float,
					   help='Maximum mel frequency')

	dist = parser.add_argument_group('distributed setup')
	dist.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0),
					  help='Rank of the process for multiproc; do not set manually')
	dist.add_argument('--world_size', type=int, default=os.getenv('WORLD_SIZE', 1),
					  help='Number of processes for multiproc; do not set manually')
	return parser


def main():
	parser = argparse.ArgumentParser(
		description='PyTorch FastPitch Training',
		allow_abbrev=False
	)
	parser = parse_args(parser)
	args, _ = parser.parse_known_args()

	if args.p_arpabet > 0.0:
		cmudict.initialize(args.cmudict_path, args.heteronyms_path)

	tf.random.set_seed(args.seed)
	np.random.seed(args.seed)
	# tf.random.set_seed(1234)
	# np.random.seed(1234)

	# Parse model specific arguments.
	parser = parse_model_args("FastPitch", parser)
	args, unk_args = parser.parse_known_args()

	if len(unk_args) > 0:
		raise ValueError(f"Invalid options {unk_args}")

	# print(json.dumps(parser))
	# print(parser)
	# print(args)

	attention_kl_loss = AttentionBinarizationLoss()
	optimizer = keras.optimizers.Adam()
	loss = FastpitchLoss(
		dur_predictor_loss_scale=args.dur_predictor_loss_scale,
		pitch_predictor_loss_scale=args.pitch_predictor_loss_scale,
		attn_loss_scale=args.attn_loss_scale
	)

	# -----------------------------------------------------------------
	# Data loading.
	text_cleaners = ['english_cleaners_v2']
	dataset_path = './ljspeech_train'
	# filelist = './filelists/ljs_audio_text_train_v3.txt'
	# filelist = './filelists/ljs_audio_text_val.txt'
	train_filelist = './filelists/ljs_audio_text_train_v3.txt'
	valid_filelist = './filelists/ljs_audio_text_val.txt'

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
	batch_size = 4#1

	train_dataset = Data(
		dataset_path, 
		train_filelist, 
		text_cleaners=text_cleaners,
		n_mel_channels=n_mel_channels,
		p_arpabet=0.0,
		n_speakers=n_speakers,
		load_mel_from_disk=True,#False,
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
	valid_dataset = Data(
		dataset_path, 
		valid_filelist, 
		text_cleaners=text_cleaners,
		n_mel_channels=n_mel_channels,
		p_arpabet=0.0,
		n_speakers=n_speakers,
		load_mel_from_disk=True,#False,
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

	train_dataset.get_max_lengths()
	train_text_max = train_dataset.max_input_len
	train_mel_max = train_dataset.max_target_len
	train_data = tf.data.Dataset.from_generator(
		train_dataset.generator,
		args=(),
		output_signature=(
			# tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
			# tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			# tf.TensorSpec(
			# 	shape=(None, n_mel_channels), dtype=tf.float32
			# ),														# mel
			# tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			# tf.TensorSpec(shape=(), dtype=tf.int64),				# text_length
			# tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# pitch
			# tf.TensorSpec(shape=(None,), dtype=tf.float32),			# energy
			# tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
			# tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# attn prior
			# tf.TensorSpec(shape=(), dtype=tf.string),				# audiopath
			tf.TensorSpec(shape=(train_text_max,), dtype=tf.int64),	# text_encoded
			tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
			tf.TensorSpec(
				shape=(train_mel_max, n_mel_channels), dtype=tf.float32
			),														# mel
			tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
			tf.TensorSpec(shape=(), dtype=tf.int64),				# text_length
			tf.TensorSpec(
				shape=(1, train_mel_max + 4), dtype=tf.float32
			),														# pitch
			tf.TensorSpec(shape=(train_mel_max,), dtype=tf.float32),# energy
			tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
			tf.TensorSpec(
				shape=(train_text_max, train_mel_max), dtype=tf.float32
			),														# attn prior
			tf.TensorSpec(shape=(), dtype=tf.string),				# audiopath
			# tf.TensorShape((batch_size, train_text_max)),
			# tf.TensorShape((batch_size,)),
			# tf.TensorShape((batch_size, train_mel_max, n_mel_channels)),
			# tf.TensorShape((batch_size,)),
			# tf.TensorShape((batch_size,)),
			# tf.TensorShape((batch_size, 1, train_mel_max)),
			# tf.TensorShape((batch_size, train_mel_max)),
			# tf.TensorShape((batch_size)),
			# tf.TensorShape((batch_size, train_text_max, train_mel_max)),
			# tf.TensorShape((batch_size,)),
		),
		# output_shapes=(
		# 	(train)
		# ),
		# output_types=(
		# 	tf.int64, tf.int64, tf.float32, tf.int64, tf.int64, 
		# 	tf.float32, tf.float32, tf.int64, tf.float32, tf.string,
		# )
	)
	# valid_data = tf.data.Dataset.from_generator(
	# 	valid_dataset.generator,
	# 	args=(),
	# 	output_signature=(
	# 		tf.TensorSpec(shape=(None,), dtype=tf.int64),			# text_encoded
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# input_lengths
	# 		tf.TensorSpec(
	# 			shape=(None, n_mel_channels), dtype=tf.float32
	# 		),														# mel
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# output_lengths
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# text_length
	# 		tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# pitch
	# 		tf.TensorSpec(shape=(None,), dtype=tf.float32),			# energy
	# 		tf.TensorSpec(shape=(), dtype=tf.int64),				# speaker id
	# 		tf.TensorSpec(shape=(None, None), dtype=tf.float32),	# attn prior
	# 		tf.TensorSpec(shape=(), dtype=tf.string),				# audiopath
	# 	),
	# )
	'''
	with tf.device('/cpu:0'): # Use CPU because GPU OOMs here (but will not when using generator). This may cost some speed.
		train_data_alt = tf.data.Dataset.from_tensor_slices(
			train_dataset.tensor_slices()
		)
		valid_data_alt = tf.data.Dataset.from_tensor_slices(
			valid_dataset.tensor_slices()
		)
	'''
	# -----------------------------------------------------------------

	# See to this link as to why I call tf.data.Dataset.batch() to see
	# batch size in the model: 
	# https://github.com/tensorflow/tensorflow/issues/43094#issuecomment-690919548
	train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Uses generator
	# train_data = train_data.batch(2)
	# train_data_alt = train_data_alt.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Uses from_tensor_slices

	model_config = get_fastpitch_config(args)
	print(json.dumps(model_config, indent=4))
	model = FastPitch(**model_config)
	model.compile(
		# optimizer=optimizer, loss=[loss, attention_kl_loss],
		optimizer=optimizer, loss=loss,
		# run_eagerly=True # Used when debugging the architecture
	)

	# model.build()
	# model.summary()
	# exit()

	model.fit(train_data, epochs=1)#, batch_size=4) # Uses generator
	# model.fit(train_data_alt, epochs=1, batch_size=4) # Uses from_tensor_slices

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
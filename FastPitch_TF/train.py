# train.py


import os
import time
import copy
import json
import argparse
from collections import defaultdict, OrderedDict
import numpy as np
import tensorflow as tf
# import models
# from fastpitch.attn_loss_function import AttentionBinarizationLoss
from fastpitch.data_function import Data
# from fastpitch.loss_function import FastPitchLoss


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

	tf.random.seed(args.seed)
	np.random.seed(args.seed)

	print(json.dumps(parser))



if __name__ == '__main__':
	main()
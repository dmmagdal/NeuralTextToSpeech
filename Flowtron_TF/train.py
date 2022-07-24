# train.py
# Train the Flowtron model.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import argparse
import json
import os
import tensorflow as tf
import ast

# from flowtron import FlowtronLoss
# from flowtron import Flowtron
from data import Data, DataCollate
# from flowtron_logger import FlowtronLogger
# from radam import RAdam


def update_params(config, params):
	for param in params:
		print(param)
		k, v = param.split("=")
		try:
			v = ast.literal_eval(v)
		except:
			print("{}:{} was not parsed".format(k, v))
			pass

		k_split = k.split('.')
		if len(k_split) > 1:
			parent_k = k_split[0]
			cur_param = ['.'.join(k_split[1:])+"="+str(v)]
			update_params(config[parent_k], cur_param)
		elif k in config and len(k_split) == 1:
			config[k] = v
		else:
			print("{}, {} params not updated".format(k, v))


def prepare_dataloaders(data_config):
	# Get data, data loaders and collate function ready
	ignore_keys = ['training_files', 'validation_files']
	trainset = Data(
		data_config['training_files'],
		**dict((k, v) for k, v in data_config.items()
		if k not in ignore_keys)
	)
	valset = Data(
		data_config['validation_files'],
		**dict((k, v) for k, v in data_config.items()
		if k not in ignore_keys), speaker_ids=trainset.speaker_ids
	)

	collate_fn = DataCollate(
		n_frames_per_step=1, use_attn_prior=trainset.use_attn_prior
	)

	train_sampler, shuffle = None, True

	# train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
	# 						  sampler=train_sampler, batch_size=batch_size,
	# 						  pin_memory=False, drop_last=True,
	# 						  collate_fn=collate_fn)

	# return train_loader, valset, collate_fn
	return trainset, valset, collate_fn



def train(n_gpus, rank, output_directory, epochs, optim_algo, 
		learning_rate, weight_decay, sigma, iters_per_checkpoint, 
		batch_size, seed, checkpoint_path, ignore_layers, 
		include_layers, finetune_layers, warmstart_checkpoint_path, 
		with_tensorboard, grad_clip_val, gate_loss, use_ctc_loss, 
		ctc_loss_weight, blank_logprob, ctc_loss_start_iter, fp16_run):
	# Set seed.
	use_ctc_loss = bool(use_ctc_loss)
	tf.random.set_seed(seed)

	# Initialize optimizer.

	# Initialize model. Resume training from checkpoints if applicable.

	# Load training and validation data.
	n_mel_channels = 80
	trainset, valset, collate_fn = prepare_dataloaders(data_config)
	#'''
	train_dataset = tf.data.Dataset.from_generator(
		trainset.generator, 
		output_signature=(
			tf.TensorSpec(
				shape=(None, n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(1,), dtype=tf.int64),
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(None, None), dtype=tf.float32)
		)
	)
	valid_dataset = tf.data.Dataset.from_generator(
		valset.generator, 
		output_signature=(
			tf.TensorSpec(
				shape=(None, n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(1,), dtype=tf.int64),
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(None, None), dtype=tf.float32)
		)
	)
	print(list(train_dataset.as_numpy_iterator())[0])
	print(list(valid_dataset.as_numpy_iterator())[0])

	max_input_length = max([tf.shape(val[2])[0] for val in train_dataset])
	print("Max text input length: {}".format(max_input_length))
	'''
	# Using the .from_tensor_slices() currently will throw an
	# error in which all the tensors for each feature must be of the
	# exact same shape. (Also as a side note: On Desktop GPU, this 
	# process will exhaust the 8GB VRAM available. Using
	# .from_generator() will NOT exhause that resource).
	train_dataset_tensors, max_input_length = trainset.generator()
	train_dataset = tf.data.Dataset.from_tensor_slices(
		train_dataset_tensors
	)
	print(max_input_length)
	print(list(train_dataset.as_numpy_iterator())[0])
	'''

	
	# Training loop.
	pass



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-p', '--params', nargs='+', default=[])
	args = parser.parse_args()
	args.rank = 0

	# Parse configs.  Globals nicer in this case
	with open(args.config) as f:
		data = f.read()

	global config
	config = json.loads(data)
	update_params(config, args.params)
	print(config)

	train_config = config["train_config"]
	global data_config
	data_config = config["data_config"]
	global dist_config
	dist_config = config["dist_config"]
	global model_config
	model_config = config["model_config"]

	# Make sure the launcher sets `RANK` and `WORLD_SIZE`.
	rank = int(os.getenv('RANK', '0'))
	n_gpus = int(os.getenv("WORLD_SIZE", '1'))
	print('> got rank {} and world size {} ...'.format(rank, n_gpus))

	if n_gpus == 1 and rank != 0:
		raise Exception("Doing single GPU training on rank > 0")

	train(n_gpus, rank, **train_config)
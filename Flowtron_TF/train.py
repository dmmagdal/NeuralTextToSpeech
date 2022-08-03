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

	# Load training and validation data. For the shape of the
	# TensorSpec for the mel-spectrograms, be aware of which dimension
	# is the n_mel_channels and which is the length of the
	# mel-spectrogram.
	n_mel_channels = 80

	with tf.device("/cpu:0"):
		trainset, valset, collate_fn = prepare_dataloaders(data_config)
		trainset.set_collate_fn(collate_fn)
		valset.set_collate_fn(collate_fn)

		#'''
		train_dataset = tf.data.Dataset.from_generator(
			trainset.generator,
			args=("training", True),
			output_signature=(
				tf.TensorSpec(
					shape=(None, n_mel_channels), dtype=tf.float32
				),
				tf.TensorSpec(shape=(1,), dtype=tf.int64),
				tf.TensorSpec(shape=(None,), dtype=tf.int64),
				tf.TensorSpec(shape=(), dtype=tf.int64),
				tf.TensorSpec(shape=(), dtype=tf.int64),
				tf.TensorSpec(shape=(None,), dtype=tf.int64),
				tf.TensorSpec(shape=(None, None), dtype=tf.float32)
			)
		)
		#'''
		valid_dataset = tf.data.Dataset.from_generator(
			valset.generator,
			args=("validation", True),
			output_signature=(
				tf.TensorSpec(
					shape=(None, n_mel_channels), dtype=tf.float32
				), # mel
				tf.TensorSpec(shape=(1,), dtype=tf.int64), # speaker
				tf.TensorSpec(shape=(None,), dtype=tf.int64), # text
				tf.TensorSpec(shape=(), dtype=tf.int64), # input_len
				tf.TensorSpec(shape=(), dtype=tf.int64), # target_len
				tf.TensorSpec(shape=(None,), dtype=tf.int64), # gate
				tf.TensorSpec(shape=(None, None), dtype=tf.float32) # attn_prior
			)
		)
		print(list(train_dataset.as_numpy_iterator())[0])
		# print(list(valid_dataset.as_numpy_iterator())[0])


	exit()

	'''
	train_dataset = tf.data.Dataset.from_generator(
		trainset.generator,
		args=("training", True),
		output_signature=(
			tf.TensorSpec(
				shape=(None, n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(1,), dtype=tf.int64),
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(None, None), dtype=tf.float32)
		)
	)
	'''
	valid_dataset = tf.data.Dataset.from_generator(
		valset.generator,
		args=("validation", True),
		output_signature=(
			tf.TensorSpec(
				shape=(None, n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(1,), dtype=tf.int64),
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(None, None), dtype=tf.float32)
		)
	)
	# print(list(train_dataset.as_numpy_iterator())[0])
	print(list(valid_dataset.as_numpy_iterator())[0])

	# Find the maximum input (text) and target (mel-spectrogram)
	# lengths for each dataset. For the target (mel-spectrogram)
	# length, be aware of which dimension is the n_mel_channels and
	# which is the length of the mel-spectrogram.
	'''
	train_max_in_len = max(
		[tf.shape(value[2])[0] for value in train_dataset]
	)
	train_max_tar_len = max(
		[tf.shape(value[0])[0] for value in train_dataset]
	)
	'''
	val_max_in_len = max(
		[tf.shape(value[2])[0] for value in valid_dataset]
	)
	val_max_tar_len = max(
		[tf.shape(value[0])[0] for value in valid_dataset]
	)
	print(f"val_max_in_len {val_max_in_len}, val_max_tar_len {val_max_tar_len}")
	# print(f"max input length {valset.get_max_lengths()[0]} max target length {valset.get_max_lengths()[1]}")
	# exit()
	'''
	# Using the .from_tensor_slices() currently will throw an
	# error in which all the tensors for each feature must be of the
	# exact same shape. (Also as a side note: On Desktop GPU, this 
	# process will exhaust the 8GB VRAM available. Using
	# .from_generator() will NOT exhause that resource).
	train_dataset_tensors, max_input_length = trainset.generator(
		"training", False
	)
	train_dataset = tf.data.Dataset.from_tensor_slices(
		train_dataset_tensors
	)
	valid_dataset_tensors, max_input_length = valset.generator(
		"validation", False
	)
	valid_dataset = tf.data.Dataset.from_tensor_slices(
		valid_dataset_tensors
	)
	print("Max text input length: {}".format(max_input_length))
	print(list(train_dataset.as_numpy_iterator())[0])
	print(list(valid_dataset.as_numpy_iterator())[0])
	'''

	# Update input and target lengths to data collate function for
	# each respective dataset. Then apply data collate function to
	# the datasets. Leverage tf.numpy_function() to wrap around the
	# call to the data collate function to allow for the input tensors
	# to be converted to numpy arrays. This allows for the tensors
	# (which are currently in graph execution mode) to be read for
	# padding. Additional information can be found in this stack
	# overflow response (https://stackoverflow.com/questions/
	# 50538038/tf-data-dataset-mapmap-func-with-eager-mode).
	'''
	collate_fn.update_max_len(train_max_in_len, train_max_tar_len)
	train_dataset = train_dataset.map(
		lambda mel, speaker, text_enc, attn_prior:
		tf.numpy_function(
			collate_fn, [mel, speaker, text_enc, attn_prior],
			[
				tf.float32, tf.int64, tf.int64, tf.int64, tf.int64, 
				tf.int64, tf.float32
			]
		), 
		num_parallel_calls=tf.data.AUTOTUNE,
	)
	'''
	collate_fn.update_max_len(val_max_in_len, val_max_tar_len)
	valid_dataset = valid_dataset.map(
		lambda mel, speaker, text_enc, attn_prior:
		tf.numpy_function(
			collate_fn, [mel, speaker, text_enc, attn_prior],
			[
				tf.float32, tf.int64, tf.int64, tf.int64, tf.int64, 
				tf.int64, tf.float32
			]
		), 
		num_parallel_calls=tf.data.AUTOTUNE,
	)
	print(list(valid_dataset.as_numpy_iterator())[0])

	# NOTE: This entire process of loading the dataset, finding the
	# maximum input (text) and target (mel-spectrogram) lengths, and
	# applying the data collate function cyles through the data 4
	# times. This means that if the training dataset takes an hour to
	# iterate through through once, then the whole process will take 4
	# hours to complete. It is advised that this loading and processing
	# is done once and saved (to be loaded again when actually training
	# or running inference).

	# Having a hard time using this to save/load dataset because:
	# 1) the load() function requires an element_spec but the 
	#	documentation on how that is supposed to look like is vague.
	# 2) on the official tensorflow documentation (currently at v2.9)
	#	it warns that tf.data.experimental.load() and 
	#	tf.data.experimental.save() are deprecated functions.
	'''
	element_spec = [
		tf.TensorSpec(
			shape=(None, n_mel_channels), dtype=tf.float32
		),
		tf.TensorSpec(shape=(1,), dtype=tf.int64),
		tf.TensorSpec(shape=(None,), dtype=tf.int64),
		tf.TensorSpec(shape=(1,), dtype=tf.int64),
		tf.TensorSpec(shape=(1,), dtype=tf.int64),
		tf.TensorSpec(shape=(None,), dtype=tf.int64),
		tf.TensorSpec(shape=(None, None), dtype=tf.float32)
	]
	tf.data.experimental.save(valid_dataset, "./LJSpeech-Valid")
	dataset = tf.data.experimental.load(
		"./LJSpeech-Valid", element_spec
	)
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
	print(json.dumps(config, indent=4))

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
# train.py
# Train the Tacotron2 model.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import copy
import time
import json
import argparse
import tensorflow as tf

from data_utils import TextMelLoader, TextMelCollate
from hparams import HParams


def prepare_dataloaders(hparams):
	# Get data, data loaders and collate function ready.
	trainset = TextMelLoader(hparams.training_files, hparams)
	valset = TextMelLoader(hparams.validation_files, hparams)
	collate_fn = TextMelCollate(n_frames_per_step=1)

	return trainset, valset, collate_fn


def train(output_dir, log_dir, checkpoint_path, hparams):
	# Set seed.

	# Initialize optimizer.

	# Initialize model. Resume training from checkpoints if applicable.

	# Load training and validation data. For the shape of the
	# TensorSpec for the mel-spectrograms, be aware of which dimension
	# is the n_mel_channels and which is the length of the
	# mel-spectrogram.
	# Using the .from_tensor_slices() currently will throw an
	# error in which all the tensors for each feature must be of the
	# exact same shape. (Also as a side note: On Desktop GPU, this 
	# process will exhaust the 8GB VRAM available. Using
	# .from_generator() will NOT exhause that resource).
	trainset, valset, collate_fn = prepare_dataloaders(hparams)

	#'''
	print("Generating training dataset...")
	train_collate_fn = copy.deepcopy(collate_fn)
	trainset.set_collate_fn(train_collate_fn)
	train_dataset = tf.data.Dataset.from_generator(
		trainset.generator,
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(), dtype=tf.int64),
			tf.TensorSpec(
				shape=(None, hparams.n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(None,), dtype=tf.float32),
			tf.TensorSpec(shape=(), dtype=tf.int64),
		)

	).batch(hparams.batch_size).prefetch(tf.data.AUTOTUNE)
	# Need to put the print or assignment statement here to allow the
	# dataset to pause before applying collate function with map()
	# later.
	# print(list(train_dataset.as_numpy_iterator())[0])
	#'''

	# Make a copy of the collate function specifically for the
	# training dataset. Update its maximum length variables. Apply
	# the collate function to the training dataset.
	# -----------------------------------------------------------
	# Leverage tf.numpy_function() to wrap around the
	# call to the data collate function to allow for the input tensors
	# to be converted to numpy arrays. This allows for the tensors
	# (which are currently in graph execution mode) to be read for
	# padding. Additional information can be found in this stack
	# overflow response (https://stackoverflow.com/questions/
	# 50538038/tf-data-dataset-mapmap-func-with-eager-mode).

	# Rearranged dataset operations to be in sequence rather than in
	# parallel. Hypothesis is that the reason why the collator function
	# is getting mixed up with the max (encoded) text length values is
	# because the parallelization of the collate function must be
	# mixing the variables.
	# UPDATE: This hypothesis turned out to not be true. The reason for
	# the collator function to mix up variables is due to the
	# multiprocessing/threading call from the train dataset still
	# running when the collate function is updated and called on the
	# validation dataset. By creating two instances of the collate
	# function, the length variables no longer mix and the collate
	# function can run successfully on each dataset.
	# Make a copy of the collate function specifically for the
	# validation dataset. Update its maximum length variables.
	print("Generating validation dataset...")
	valid_collate_fn = copy.deepcopy(collate_fn)
	valset.set_collate_fn(valid_collate_fn)
	valid_dataset = tf.data.Dataset.from_generator(
		valset.generator,
		output_signature=(
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(), dtype=tf.int64),
			tf.TensorSpec(
				shape=(None, hparams.n_mel_channels), dtype=tf.float32
			),
			tf.TensorSpec(shape=(None,), dtype=tf.int64),
			tf.TensorSpec(shape=(), dtype=tf.int64),
		)
	).batch(hparams.batch_size).prefetch(tf.data.AUTOTUNE)
	# print(list(valid_dataset.as_numpy_iterator())[0])

	# print(list(train_dataset.as_numpy_iterator())[0])
	# print(list(valid_dataset.as_numpy_iterator())[0])


	
	# Functions to save processed dataset as TFRecords. Currently, the
	# processing of the LJSpeech dataset for Tacotron2 is significantly
	# faster than for Flowtron. Given that this was originally intended
	# for the Flowtron model which takes 2.5 to 3 hours vs 5 to 15
	# minutes for Tacotron2, there is no need to uncomment this code.
	'''
	def _float_feature(value):
		# Return a float_list from a float.
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))


	def _int64_feature(value):
		# Return an int64_list from an int.
		return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


	def serialize_example(mels, text, input_len, output_len, gate):
		# Create a dictionary mapping the feature name to the
		# tf.train.Example-compatible data type.
		feature = {
			"mels": _float_feature(mels),
			"text": _int64_feature(text), 
			"input_len": _int64_feature(input_len), 
			"output_len": _int64_feature(output_len), 
			"gate": _int64_feature(gate),
		}

		# Create a Features message using tf.train.Example.
		example_proto = tf.train.Example(
			features=tf.train.Features(feature=feature)
		)
		return example_proto.SerializeToString()


	def write_tfrecords(name, dataset):
		path = "./dataset_tfrecords/" + name + "/"
		writer = tf.io.TFRecordWriter(path)
		dataset_list = list(dataset.as_numpy_iterator())
		for idx in range(len(dataset_list)):
			text, input_len, mels, gate, output_len = dataset_list[idx]
			example = serialize_example(
				text, input_len, mels, gate, output_len 
			)
			writer.write(example)


	write_tfrecords("LJSpeech_train", train_dataset)
	write_tfrecords("LJSpeech_valid", valid_dataset)
	'''


	exit()
	
	# Training loop.
	pass



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument('-c', '--config', type=str,
	# 					help='JSON file for configuration')
	# parser.add_argument('-p', '--params', nargs='+', default=[])
	args = parser.parse_args()
	# args.rank = 0

	# Parse configs. Globals nicer in this case
	# with open(args.config) as f:
	# 	data = f.read()

	# global config
	# config = json.loads(data)
	# print(json.dumps(config, indent=4))

	output_dir = "./tacotron2_ljspeech_train_output"
	log_dir = "./tacotron2_ljspeech_train_log"
	checkpoint_path = "./tacotron2_ljspeech_train_checkpoint"
	hparams = HParams()

	train(output_dir, log_dir, checkpoint_path, hparams)
# pitch_transform.py


import tensorflow as tf


def pitch_transform_custom(pitch, pitch_lens):
	# Apply a custom pitch transformation to predicted pitch values.
	# This simple modification linearly increases the pitch throughout
	# the utterance from 0.5 of predicted pitch to 1.5 of predicted
	# pitch. In other words, it starts low and ends high.
	# @param: pitch, (tf.Tensor of shape (batch_size, max_len))
	#	predicted pitch values for each lexical unit, padded to max_len
	#	(in Hz).
	# @param: pitch_lens (tf.Tensor of shape (batch_size, max_len))
	#	number of lexical units in each utterance.
	# @return: returns (tf.Tensor) modified pitch (in Hz).
	weights = tf.range(pitch.shape[1], dtype=tf.float32)

	# The weights increase linearly from 0.0 to 1.0 in every i-th row
	# in the range (0, pitch_lens[i]).
	weights = tf.expand_dims(weights, axis=0) /\
		tf.expand_dims(pitch_lens, axis=1)

	# Shift the range from (0.0, 1.0) to (0.5, 1.5).
	weights += 0.5

	return pitch * weights
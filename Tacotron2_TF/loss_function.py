# loss_function.py


import tensorflow as tf
from tensorflow import keras


def Tacotron2Loss(model_output, targets):
	mel_target, gate_target = targets[0], targets[1]
	gate_target = tf.reshape(gate_target, [-1, 1])

	mel_out, mel_out_postnet, get_out, _ = model_output
	gate_out = tf.reshape(gate_out, [-1, 1])
	mel_loss = keras.losses.MeanSquaredError()(mel_out, mel_target) +\
		keras.losses.MeanSquaredError()(mel_out_postnet, mel_target)
	gate_loss = keras.losses.BinaryCrossEntropy(from_logits=True)(
		gate_out, gate_target
	)
	return mel_loss + gate_loss
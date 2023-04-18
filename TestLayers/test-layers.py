# test-layers.py
# This is a test file for tracking the general shape of inputs and
# their lengths for the Grad-TTS TextEncoder module (particularly, the
# beginning components with the ConvReLUNorm Prenet).
# Tensorflow 2.7.0
# Python 3.7
# Windows/MacOS/Linux


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mask_from_lens(lengths):
	max_len = tf.math.reduce_max(lengths).numpy().item()
	mask = tf.sequence_mask(lengths, max_len, dtype=tf.bool)
	return mask


# OpenAI ChatGPT implementation of ConvReLUNorm layer.
class ConvReluNorm(layers.Layer):
	def __init__(self, in_channels, hidden_channels, kernel_size, 
			n_layers, p_dropout, **kwargs):
		super(ConvReluNorm, self).__init__(**kwargs)
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.n_layers = n_layers
		self.p_dropout = p_dropout

		self.conv_layers = []
		self.norm_layers = []
		self.conv_layers.append(
			layers.Conv1D(
				self.hidden_channels, self.kernel_size, padding='same'
			)
		)
		self.norm_layers.append(layers.LayerNormalization())
		self.relu_drop = keras.Sequential(
			[layers.ReLU(), layers.Dropout(self.p_dropout)]
		)
		for _ in range(self.n_layers - 1):
			self.conv_layers.append(
				layers.Conv1D(
					self.hidden_channels, self.kernel_size, padding='same'
				)
			)
			self.norm_layers.append(layers.LayerNormalization())

	def call(self, inputs, x_mask):
		# x_org, = inputs
		x_org = inputs # I modified the above line because it didnt need to unpack values
		for i in range(self.n_layers):
			x = self.conv_layers[i](x_org * x_mask)
			x = self.norm_layers[i](x)
			x = self.relu_drop(x)
		return x * x_mask

# Additional notes from ChatGPT on ConvReLUNorm layer:
# Here we create a ConvReluNorm class that inherits from layers.Layer. The init 
# method initializes the parameters such as in_channels, hidden_channels, 
# kernel_size, n_layers and p_dropout, and also creates the convolution layers, 
# normalization layers and relu + dropout layers. The call method takes in inputs 
# and x_mask as arguments and performs the convolution, normalization and relu + 
# dropout operations. This can be used as a regular layer in a keras model.
#
# We can then call this layer in a sequential model as follows:
# ```
# model = tf.keras.Sequential()
# model.add(
# 	ConvReluNorm(
# 		in_channels=10, hidden_channels=32, kernel_size=3, n_layers=4, p_dropout=0.1
# 	)
# )
# ```
#
# Note that here we expect the input tensors to be passed to this layer as a tuple 
# (x_org,) and the x_mask as a separate argument, which is used to mask the input.
# 
# The shape of the inputs for a ConvReLuNorm layer implemented in Tensorflow would 
# be a tuple (x_org,), with x_org being a tensor of shape (batch_size, 
# sequence_length, number_of_channels). Additionally, the layer expects a separate 
# argument x_mask of shape (batch_size, sequence_length) representing the mask to be 
# applied to the input.


# Data & lengths. batch_size is 2.
x = tf.convert_to_tensor(
	[
		[133, 50, 28, 30, 19, 49, 573, 20, 60], 
		[1, 4, 2, 57, 37, 471, 0, 0, 0]
	]
)
x_lengths = tf.convert_to_tensor(
	[9, 6]
)

# Mask
x_mask = mask_from_lens(x_lengths)

# parameters
n_vocab = 573
n_enc_channels = 192
filter_channel = 768
filter_channel_dp = 256
kernel_size = 5
n_layers = 3
p_dropout = 0.5

embedding = layers.Embedding(n_vocab, n_enc_channels)
convReLuNorm = ConvReluNorm(
	filter_channel, filter_channel, kernel_size=kernel_size,
	n_layers=n_layers, p_dropout=p_dropout
)

x = embedding(x) + math.sqrt(n_enc_channels)
# x_mask = tf.cast(tf.expand_dims(x_mask, 1), dtype=x.dtype) # Incompatible shape matmul error in convReLuNorm layer.
# ChatGPT says to expand_dims across the last dimension to successfully 
# multiply x and x_mask
x_mask = tf.cast(tf.expand_dims(x_mask, -1), dtype=x.dtype)
# x_mask = tf.cast(tf.expand_dims(x_mask, 2), dtype=x.dtype) # This also works too.
print(f"x {x}, {x.shape}")
print(f"x_mask {x_mask}, {x_mask.shape}")
# ChatGPT entry:
#
# Q > I want to multiply x (shape (2, 9, 192)) and y (shape (2, 9)) in tensorflow 
# such that the result is of shape (2, 9, 192)?
#
# A > In Tensorflow, you can use broadcasting to perform the multiplication between 
# x and y, even if their shapes don't match exactly. The resulting shape will be the 
# broadcasted shape of the two tensors. To get the desired output shape of (2, 9, 
# 192), you can use the following:
# ```
# result = x * tf.expand_dims(y, axis=-1)
# ```
# This will expand the last dimension of y from shape (2, 9) to (2, 9, 1) using 
# tf.expand_dims() so that it can be broadcasted to match the last dimension of x. The 
# resulting tensor will have the shape (2, 9, 192).
# Alternatively, you can use tf.broadcast_to() method to broadcast y to match the shape 
# of x.
# ```
# result = x * tf.broadcast_to(tf.expand_dims(y, axis=-1), x.shape)
# ```
# It will also give the same result tensor with shape (2, 9, 192)
# You can also use tf.tile to repeat y along the third dimension of x
# ```
# result = x * tf.tile(tf.expand_dims(y, axis=-1), [1, 1, x.shape[2]])
# ```
# This will give you the result tensor with shape (2, 9, 192)

print(f"embedding x {x}, {x.shape}, {x.dtype}")
conv_relu_norm_outs = convReLuNorm(x, x_mask)
print(f"conv_relu_norm_outs: {conv_relu_norm_outs}, {conv_relu_norm_outs.shape}")

# In this comment block, there is documentation on implementing a 
# multihead attention from scratch. This source is outside of the ones 
# provided by Tensorflow their official examples/documentation.
# Source: https://medium.com/@pranavjadhav001/implementing-multi-head-
#	self-attention-layer-using-tensorflow-e19c8fc7887
# TF Source1: https://www.tensorflow.org/text/tutorials/transformer
# TF Source2: https://blog.tensorflow.org/2019/05/transformer-chatbot-
#	tutorial-with-tensorflow-2.html
# TF Source3: https://www.tensorflow.org/api_docs/python/tf/keras/
#	layers/MultiHeadAttention
# TF Source4: https://www.tensorflow.org/api_docs/python/tf/keras/
#	layers/EinsumDense
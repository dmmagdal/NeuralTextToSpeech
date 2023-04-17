# nn_utils.py
# The following are utility layers that wrap around existing layers
# with the model. These are made to emulate pytorch's weight_norm() &
# spectral_norm() functions from torch.nn.utils. There was also 
# supposed to be a layer for the remove_weight_norm() function as well
# but ChatGPT could not come up with anything for that.


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Curtesy of ChatGPT.
class WeightNorm(layers.Layer):
	def __init__(self, layer):
		super(WeightNorm, self).__init__()
		self.layer = layer
		
	
	def build(self, input_shape):
		self.layer.build(input_shape)
		self.w = self.add_weight(
			name='w', shape=self.layer.kernel.shape,
			initializer='glorot_uniform', trainable=True
		)
		self.g = self.add_weight(
			name='g', shape=(self.layer.kernel.shape[-1],),
			initializer='ones', trainable=True
		)


	def call(self, inputs, training=None):
		if training:
			self.layer.kernel = self.w * tf.math.l2_normalize(
				self.layer.kernel, axis=[0, 1, 2]
			)
			self.layer.bias = self.layer.bias * self.g
			return self.layer(inputs)
		else:
			return self.layer(inputs)
		

class SpectralNorm(layers.Layer):
	def __init__(self, layer, power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.layer = layer
		self.power_iterations = power_iterations


	def build(self, input_shape):
		self.layer.build(input_shape)
		self.u = self.add_weight(
			name='u', shape=(1, self.layer.kernel.shape[-1]),
			initializer='glorot_uniform', trainable=False
		)


	def call(self, inputs):
		w = self.layer.kernel
		w_shape = w.shape.as_list()
		w = tf.reshape(w, [-1, w_shape[-1]])
		u = self.u

		for _ in range(self.power_iterations):
			v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
			u = tf.math.l2_normalize(tf.matmul(v, w))

		sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))[0, 0]
		self.u.assign(u)

		return self.layer(inputs) / sigma
	

# A quick note on what spectral normalization is (also curtesy of
# ChatGPT):
# Spectral normalization is a technique for constraining the Lipschitz 
# constant of a neural network. The Lipschitz constant measures the rate 
# of change of a function, and it plays a crucial role in ensuring the 
# stability of the network during training. A high Lipschitz constant can
# cause problems such as vanishing or exploding gradients, which can 
# prevent the network from converging to a good solution.
#
# In spectral normalization, the weights of each layer are normalized 
# such that the spectral norm (i.e., the largest singular value) of the 
# weight matrix is constrained to a fixed value. This has the effect of 
# limiting the Lipschitz constant of the layer, which can help to 
# prevent instability during training.
#
# Spectral normalization was introduced in the paper "Spectral 
# Normalization for Generative Adversarial Networks" by Miyato et al. 
# (2018), and it has since been applied to a wide range of neural 
# network architectures.

# Note to self: The remove_weight_norm() is only used on ResBlocks 
# (both ResBlock1 & ResBlock2) which are exclusive to the Generator 
# architecture of the GAN. The ResBlocks are also only use 
# weight_norm(). The remove_weight_norm() seems to only be called
# during inference for the Generator, which means that the above
# code can be modified to run only during training. spectral_norm() is
# called potentially within the DiscriminatorP & DiscriminatorS
# architectures of the GAN. Unlike weight_norm() being connected to
# remove_weight_norm(), spectral_norm() has no partner function.
# diffusion-unet-shapes.py
# Experiment with the shape of different data coming into the building 
# blocks of the diffusion unet that is a part of GradTTS.


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import tensorflow_addons as tfa
from tensorflow_addons.layers import GroupNormalization


class Mish(layers.Layer):
	def __init__(self):
		super(Mish, self).__init__()
		self.softplus = layers.Activation('softplus')


	def call(self, x):
		return x * tf.math.tanh(self.softplus(x))
	

class Block(layers.Layer):
	def __init__(self, dim_out, groups=8):
		super(Block, self).__init__()
		self.block = Sequential([
			layers.Conv2D(dim_out, 3, padding="same"),
			GroupNormalization(groups=groups),
			Mish()
		])


	def call(self, x, mask):
		output = self.block(x * mask)
		return output * mask
	
# Original (Pytorch shapes)
x = tf.random.normal((16, 2, 80, 172))
mask = tf.cast(tf.random.normal((16, 1, 1, 172)), dtype=tf.bool)

# Option 1 shapes (transpose second dim to the last)
x = tf.random.normal((16, 80, 172, 2))
mask = tf.cast(tf.random.normal((16, 1, 172, 1)), dtype=tf.bool)
block = Block(64)

output = block(x, tf.cast(mask, dtype=tf.float32))
print(output.shape)
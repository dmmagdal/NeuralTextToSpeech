# model.py


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers
from utils import get_padding


LRELU_SLOPE = 0.1


class ResBlock1(layers.Layer):
	def __init__(self, hparams, channels, kernel_size=3, dilation=(1, 3, 5)):
		super(ResBlock1, self).__init__()
		self.hparams = hparams
		self.convs1 = [
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=dilation[0],
				padding=get_padding(kernel_size, dilation[0])
			),
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=dilation[1],
				padding=get_padding(kernel_size, dilation[1])
			),
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=dilation[2],
				padding=get_padding(kernel_size, dilation[2])
			),
		]
		self.convs2 = [
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=1,
				padding=get_padding(kernel_size, 1)
			),
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=1,
				padding=get_padding(kernel_size, 1)
			),
			layers.Conv1D(
				channels, kernel_size, strides=1, dilation=1,
				padding=get_padding(kernel_size, 1)
			),
		]
		
	def call(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			xt = layers.LeakyReLU(LRELU_SLOPE)(x)
			xt = c1(xt)
			xt = layers.LeakyReLU(LRELU_SLOPE)(x)
			xt = c2(xt)
			x = xt + x
		return x
	

class ResBlock2(layers.Layer):
	def __init__(self):
		super(ResBlock2, self).__init__()


class Generator(layers.Layer):
	def __init__(self):
		super(Generator, self).__init__()


class DiscriminatorP(layers.Layer):
	def __init__(self):
		super(DiscriminatorP, self).__init__()


class MultiPeriodDiscriminator(layers.Layer):
	def __init__(self):
		super(MultiPeriodDiscriminator, self).__init__()


class DiscriminatorS(layers.Layer):
	def __init__(self):
		super(DiscriminatorS, self).__init__()


class MultiScaleDiscriminator(layers.Layer):
	def __init__(self):
		super(MultiScaleDiscriminator, self).__init__()
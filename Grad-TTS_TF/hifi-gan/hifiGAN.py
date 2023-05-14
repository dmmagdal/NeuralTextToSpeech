# hifiGAN.py


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xutils import get_padding


LRELU_SLOPE = 0.1


class ResBlock1(layers.Layer):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
		super(ResBlock1, self).__init__()
		self.h = h
		self.convs1 = [
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[0], 
				padding=get_padding(kernel_size, dilation[0])
			),
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[1], 
				padding=get_padding(kernel_size, dilation[1])
			),
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[2], 
				padding=get_padding(kernel_size, dilation[2])
			),
		]
		self.leakyrelu1 = layers.LeakyReLU(LRELU_SLOPE)
		self.convs2 = [
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[0], 
				padding=get_padding(kernel_size, dilation[0])
			),
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[1], 
				padding=get_padding(kernel_size, dilation[1])
			),
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[2], 
				padding=get_padding(kernel_size, dilation[2])
			),
		]
		self.leakyrelu2 = layers.LeakyReLU(LRELU_SLOPE)


	def call(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			xt = self.leakyrelu1(x)
			xt = c1(xt)
			xt = self.leakyrelu2(xt)
			xt = c2(xt)
			x = xt + x
		return x


class ResBlock2(layers.Layer):
	def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
		super(ResBlock1, self).__init__()
		self.h = h
		self.convs = [
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[0], 
				padding=get_padding(kernel_size, dilation[0])
			),
			layers.Conv1D(
				channels, kernel_size=kernel_size, strides=1,
				dilation_rate=dilation[1], 
				padding=get_padding(kernel_size, dilation[1])
			),
		]
		self.leakyrelu = layers.LeakyReLU(LRELU_SLOPE)


	def call(self, x):
		for c in self.convs:
			xt = self.leakyrelu(x)
			xt = c(xt)
			x = xt + x
		return x


class Generator(layers.Layer):
	def __init__(self, h):
		super(Generator, self).__init__()
		self.h = h
		self.num_kernels = len(h.resblock_kernel_sizes)
		self.num_upsamples = len(h.upsample_rates)
		self.conv_pre = layers.Conv1D(
			h.upsample_initial_channel, kernel_size=7, strides=1,
			padding="causal"
		)
		resblock = ResBlock1 if h.resblock == '1' else ResBlock2

		self.ups = []
		for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
			self.ups.append(
				layers.Conv1DTranspose(
					h.upsample_initial_channel // (2 ** (i + 1)),
					kernel_size=k, strides=u,
					padding="causal" if ((k - u) // 2) else "same"
				)
			)

		self.resblocks = []
		for i in range(len(self.ups)):
			ch = h.upsample_initial_channel // (2 ** (i + 1))
			for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
				self.resblocks.append(resblock(h, ch, k, d))

		self.conv_post = layers.Conv1D(
			1, kernel_size=7, strides=1, padding="same"
		)
		self.leakyrelu1 = layers.LeakyReLU(LRELU_SLOPE)
		self.leakyrelu2 = layers.LeakyReLU()


	def call(self, x):
		x = self.conv_pre(x)
		for i in rante(self.num_upsamples):
			x = self.leakyrelu1(x)
			x = self.ups(x)
			xs = None
			for j in range(self.num_kernels):
				if xs is None:
					xs = self.resblocks[i * self.num_kernels + j](x)
				else:
					xs += resblocks[i * self.num_kernels + j](x)
			x = xs / self.num_kernels
		x = self.leakyrelu2(x)
		x = self.conv_post(x)
		x = tf.math.tanh(x)

		return x


class DiscriminatorP(layers.Layer):
	def __init__(self, period, kernel_size=5, strides=3, use_spectral_norm=False):
		super(DiscriminatorP, self).__init__()
		self.period = period
		# norm_f = weight_norm if use_spectral_norm == False else spectral_norm
		self.convs = [
			layers.Conv2D(
				32, kernel_size=(kernel_size, 1), strides=(strides, 1),
				padding=(get_padding(5, 1), 0)
			),
			layers.Conv2D(
				128, kernel_size=(kernel_size, 1), strides=(strides, 1),
				padding=(get_padding(5, 1), 0)
			),
			layers.Conv2D(
				512, kernel_size=(kernel_size, 1), strides=(strides, 1),
				padding=(get_padding(5, 1), 0)
			),
			layers.Conv2D(
				1024, kernel_size=(kernel_size, 1), strides=(strides, 1),
				padding=(get_padding(5, 1), 0)
			),
			layers.Conv2D(
				1024, kernel_size=(kernel_size, 1), strides=1,
				padding="valid"
			),
		]
		self.conv_post = layers.Conv2D(
			1, kernel_size=(3, 1), strides=1, padding="valid"
		)
		self.leakyrelu = layers.LeakyReLU(LRELU_SLOPE)
		self.flatten = layers.Flatten()


	def call(self, x):
		fmap = []

		# 1D to 2D
		b, t, c = tf.shape(x)
		if t % self.period != 0:
			n_pad = self.period - (t % sef.period)
			x = tf.pad(x, [0, n_pad], mode="REFLECT")
			t = t + n_pad
		x = tf.reshape(x, [b, t // self.period, self.period, c])

		for l in self.convs:
			x = l(x)
			x = self.leakyrelu(x)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		x = self.flatten(x)
		return x, fmap




def feature_loss(fmap_r, fmap_g):
	loss = 0
	for dr, dg in zip(fmap_r, fmap_g):
		for rl, gl in zip(dr, dg):
			loss += tf.math.reduce_mean(tf.math.abs(rl - gl))

	return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
	loss = 0
	r_losses = []
	g_losses = []
	for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
		r_loss = tf.math.reduce_mean((1 - dr) ** 2)
		g_loss = tf.math.reduce_mean(dg ** 2)
		loss += (r_loss + g_loss)
		r_losses.append(r_loss.numpy().item())
		g_losses.append(g_loss.numpy().item())

	return loss, r_losses, g_losses


def generator_loss(disc_outputs):
	loss = 0
	gen_losses = []
	for dg in disc_outputs:
		l = tf.math.reduce_mean((1 - dg) ** 2)
		gen_losses.append(l)
		loss += l

	return loss, gen_losses
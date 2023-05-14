# model.py


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.layers import SpectralNormalization
from utils import get_padding
from nn_utils import WeightNorm, SpectralNorm


LRELU_SLOPE = 0.1

# TODO: 
# 1) Add weight initialization to key layers


class ResBlock1(layers.Layer):
	def __init__(self, hparams, channels, kernel_size=3, dilation=(1, 3, 5)):
		super(ResBlock1, self).__init__()
		self.hparams = hparams
		self.convs1 = [
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, 
					dilation_rate=dilation[0],
					padding=get_padding(kernel_size, dilation[0])
				), data_init=False
			),
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, 
					dilation_rate=dilation[1],
					padding=get_padding(kernel_size, dilation[1])
				), data_init=False
			),
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, 
					dilation_rate=dilation[2],
					padding=get_padding(kernel_size, dilation[2])
				), data_init=False
			),
		]
		self.convs2 = [
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, dilation_rate=1,
					padding=get_padding(kernel_size, 1)
				), data_init=False
			),
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, dilation_rate=1,
					padding=get_padding(kernel_size, 1)
				), data_init=False
			),
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, dilation_rate=1,
					padding=get_padding(kernel_size, 1)
				), data_init=False
			),
		]
		self.convs1LReLU = layers.LeakyReLU(LRELU_SLOPE)
		self.convs2LReLU = layers.LeakyReLU(LRELU_SLOPE)


	def call(self, x):
		for c1, c2 in zip(self.convs1, self.convs2):
			# xt = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			xt = self.convs1LReLU(x)
			xt = c1(xt)
			# xt = layers.LeakyReLU(LRELU_SLOPE)(xt) # Original
			xt = self.convs2LReLU(xt)
			xt = c2(xt)
			x = xt + x
		return x
	

class ResBlock2(layers.Layer):
	def __init__(self, hparams, channels, kernel_size=3, dilation=(1, 3)):
		super(ResBlock2, self).__init__()
		self.hparams = hparams
		self.convs = [
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, 
					dilation_rate=dilation[0],
					padding=get_padding(kernel_size, dilation[0])
				), data_init=False
			),
			tfa.layers.WeightNormalization(
				layers.Conv1D(
					channels, kernel_size, strides=1, 
					dilation_rate=dilation[1],
					padding=get_padding(kernel_size, dilation[1])
				), data_init=False
			),
		]
		self.convLReLU = layers.LeakyReLU(LRELU_SLOPE)


	def call(self, x):
		for c in self.convs:
			# xt = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			xt = self.convLReLU(x)
			xt = c(xt)
			x = xt + x
		return x


class Generator(keras.Model):
	def __init__(self, hparams):
		super(Generator, self).__init__()
		self.hparams = hparams
		self.num_kernels = len(hparams.resblock_kernel_sizes)
		self.num_upsamples = len(hparams.upsample_rates)
		self.conv_pre = tfa.layers.WeightNormalization(
			layers.Conv1D(
				hparams.upsample_initial_channel, 7, 1, padding="same"#padding=3
			), data_init=False
		)
		resblock = ResBlock1 if hparams.resblock == '1' else ResBlock2

		self.ups = []
		for i, (u, k) in enumerate(zip(hparams.upsample_rates, hparams.upsample_kernel_sizes)):
			self.ups.append(
				tfa.layers.WeightNormalization(
					layers.Conv1DTranspose(
						hparams.upsample_initial_channel // (2 ** (i + 1)),
						kernel_size=k, strides=u, padding="same"#padding=(k - u) // 2
					), data_init=False
				)
			)

		self.resblocks = []
		for i in range(len(self.ups)):
			ch = hparams.upsample_initial_channel // (2 ** (i + 1))
			for j, (k, d) in enumerate(zip(hparams.resblock_kernel_sizes, hparams.resblock_dilation_sizes)):
				self.resblocks.append(resblock(hparams, ch, k, d))

		self.conv_post = tfa.layers.WeightNormalization(
			layers.Conv1D(
				1, kernel_size=7, strides=1, padding="same"#padding=3
			), data_init=False
		)

		self.upLeakyReLU = layers.LeakyReLU(LRELU_SLOPE)
		self.leakyReLU = layers.LeakyReLU()

	
	def call(self, x):
		x = self.conv_pre(x)
		for i in range(self.num_upsamples):
			# x = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			x = self.upLeakyReLU(x)
			x = self.ups[i](x)
			xs = None
			for j in range(self.num_kernels):
				if xs is None:
					xs = self.resblocks[i * self.num_kernels + j](x)
				else:
					xs += self.resblocks[i * self.num_kernels + j](x)
			x = xs / self.num_kernels
		# x = layers.LeakyReLU()(x) # Original
		x = self.leakyReLU(x)
		x = self.conv_post(x)
		x = tf.math.tanh(x)

		return x


class DiscriminatorP(layers.Layer):
	def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
		super(DiscriminatorP, self).__init__()
		self.period = period
		# norm_f = weight_norm if use_spectral_norm == False else spectral_norm # Need to figure out spectral & weight norm functions.
		# norm_f = WeightNorm if use_spectral_norm == False else SpectralNorm
		# norm_f = tfa.layers.WeightNormalization if use_spectral_norm == False else tfa.layers.SpectralNormalization
		norm_f = WeightNormalization if use_spectral_norm == False else SpectralNormalization
		second_arg = False if use_spectral_norm == False else 1
		self.convs = [
			norm_f(
				layers.Conv2D(
					32, (kernel_size, 1), (stride, 1), padding="same"#padding=(get_padding(5, 1), 0)
				), second_arg
			),
			norm_f(
				layers.Conv2D(
					128, (kernel_size, 1), (stride, 1), padding="same"#padding=(get_padding(5, 1), 0)
				), second_arg
			),
			norm_f(
				layers.Conv2D(
					512, (kernel_size, 1), (stride, 1), padding="same"#padding=(get_padding(5, 1), 0)
				), second_arg
			),
			norm_f(
				layers.Conv2D(
					1024, (kernel_size, 1), (stride, 1), padding="same"#padding=(get_padding(5, 1), 0)
				), second_arg
			),
			norm_f(
				layers.Conv2D(
					1024, (kernel_size, 1), 1, padding="same"#padding=(2, 0)
				), second_arg
			),
		]
		self.conv_post = norm_f(
			layers.Conv2D(1, (3, 1), 1, padding="same"),#padding=(1, 0)), 
			second_arg
		)

		self.convsLeakyReLU = layers.LeakyReLU(LRELU_SLOPE)
		self.flatten = layers.Flatten()


	def call(self, x):
		fmap = []

		# 1d to 2d
		# b, c, t = x.shape # Original in Pytorch
		b, t, c = x.shape # Tensorflow ordering
		if t % self.period != 0: # pad first
			n_pad = self.period - (t % self.period)
			# x = F.pad(x, (0, n_pad), "reflect")
			# x = tf.pad(x, [[0, n_pad]], "REFLECT") 
			x = tf.pad(x, [[0, 0], [0, n_pad], [0, 0]], "REFLECT") # Updated for appropriate number of dims
			t = t + n_pad
		# x = x.view(b, c, t // self.period, self.period)
		x = tf.reshape(x, [b, t // self.period, self.period, c])

		for l in self.convs:
			x = l(x)
			# x = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			x = self.convsLeakyReLU(x)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		# x = torch.flatten(x, 1, -1)
		x = self.flatten(x)

		return x, fmap


# class MultiPeriodDiscriminator(layers.Layer):
class MultiPeriodDiscriminator(keras.Model):
	def __init__(self):
		super(MultiPeriodDiscriminator, self).__init__()
		self.discriminators = [
			DiscriminatorP(2),
			DiscriminatorP(3),
			DiscriminatorP(5),
			DiscriminatorP(7),
			DiscriminatorP(11),
		]


	def call(self, x):
		y, y_hat = x
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for _, d in enumerate(self.discriminators):
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(layers.Layer):
	def __init__(self, use_spectral_norm=False):
		super(DiscriminatorS, self).__init__()
		# norm_f = WeightNorm if use_spectral_norm == False else SpectralNorm
		# norm_f = tfa.layers.WeightNormalization if use_spectral_norm == False else tfa.layers.SpectralNormalization
		norm_f = WeightNormalization if use_spectral_norm == False else SpectralNormalization
		second_arg = False if use_spectral_norm == False else 1
		self.convs = [
			norm_f(
				layers.Conv1D(
					128, kernel_size=15, strides=1, padding="valid" #padding=7 # valid padding is same, valid, or causal
				), second_arg
			),
			norm_f(
				layers.Conv1D(
					128, kernel_size=41, strides=2, groups=4, 
					padding="same" #padding=20 # valid padding is same, valid, or causal
				), second_arg
			),
			norm_f(
				layers.Conv1D(
					256, kernel_size=41, strides=2, groups=16, 
					padding="same" #padding=20 # valid padding is same, valid, or causal
				), second_arg
			),
			norm_f(
				layers.Conv1D(
					512, kernel_size=41, strides=4, groups=16, 
					padding="same" #padding=20 # valid padding is same, valid, or causal
				), second_arg
			),
			norm_f(
				layers.Conv1D(
					1024, kernel_size=41, strides=4, groups=16, 
					padding="same" #padding=20 # valid padding is same, valid, or causal
				), second_arg
			),
			norm_f(
				layers.Conv1D(
					1024, kernel_size=41, strides=1, groups=16, 
					padding="same" #padding=20 # valid padding is same, valid, or causal
				), second_arg

			),
			norm_f(
				layers.Conv1D(
					1024, kernel_size=5, strides=1, padding="same" #padding=2 # valid padding is same, valid, or causal
				), second_arg
			),
		]
		self.conv_post = norm_f(
			layers.Conv1D(
				1, kernel_size=3, strides=1, padding="same" #padding=1 # valid padding is same, valid, or causal
			), second_arg
		)
		self.convsLeakyReLU = layers.LeakyReLU(LRELU_SLOPE)
		self.flatten = layers.Flatten()


	def call(self, x):
		fmap = []

		for l in self.convs:
			x = l(x)
			# x = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			x = self.convsLeakyReLU(x)
			fmap.append(x)
		x = self.conv_post(x)
		fmap.append(x)
		# x = torch.flatten(x, 1, -1)
		x = self.flatten(x)

		return x, fmap


class MultiScaleDiscriminator(keras.Model):
	def __init__(self):
		super(MultiScaleDiscriminator, self).__init__()
		self.discriminators = [
			DiscriminatorS(use_spectral_norm=True),
			DiscriminatorS(),
			DiscriminatorS(),
		]
		self.mean_pools = [
			layers.AveragePooling1D(
				pool_size=4, strides=2, padding="same" #padding=2 # valid padding is same or valid
			),
			layers.AveragePooling1D(
				pool_size=4, strides=2, padding="same" #padding=2 # valid padding is same or valid
			),
		]


	def call(self, x):
		y, y_hat = x
		y_d_rs = []
		y_d_gs = []
		fmap_rs = []
		fmap_gs = []
		for i, d in enumerate(self.discriminators):
			if i != 0:
				y = self.mean_pools[i - 1](y)
				y_hat = self.mean_pools[i - 1](y_hat)
			y_d_r, fmap_r = d(y)
			y_d_g, fmap_g = d(y_hat)
			y_d_rs.append(y_d_r)
			fmap_rs.append(fmap_r)
			y_d_gs.append(y_d_g)
			fmap_gs.append(fmap_g)

		return y_d_rs, y_d_gs, fmap_rs, fmap_gs



def get_generator(input_shape, hparams):
		num_kernels = len(hparams.resblock_kernel_sizes)
		num_upsamples = len(hparams.upsample_rates)
		conv_pre = WeightNormalization(
			layers.Conv1D(
				hparams.upsample_initial_channel, 7, 1, #padding=3
			), data_init=False
		)
		resblock = ResBlock1 if hparams.resblock == '1' else ResBlock2

		ups = []
		for i, (u, k) in enumerate(zip(hparams.upsample_rates, hparams.upsample_kernel_sizes)):
			ups.append(
				WeightNormalization(
					layers.Conv1DTranspose(
						hparams.upsample_initial_channel // (2 ** (i + 1)),
						kernel_size=k, strides=u, padding="same"#padding=(k - u) // 2
					)
				), data_init=False
			)

		resblocks = []
		for i in range(len(ups)):
			ch = hparams.upsample_initial_channel // (2 ** (i + 1))
			for j, (k, d) in enumerate(zip(hparams.resblock_kernel_sizes, hparams.resblock_dilation_sizes)):
				resblocks.append(resblock(hparams, ch, k, d))

		conv_post = WeightNormalization(
			layers.Conv1D(
				1, kernel_size=7, strides=1, padding="same"#padding=3
			), data_init=False
		)

		upLeakyReLU = layers.LeakyReLU(LRELU_SLOPE)
		leakyReLU = layers.LeakyReLU()

		inp = layers.Input(input_shape)
		# x = conv_pre(x)
		x = conv_pre(inp)
		for i in range(num_upsamples):
			# x = layers.LeakyReLU(LRELU_SLOPE)(x) # Original
			x = upLeakyReLU(x)
			x = ups[i](x)
			xs = None
			for j in range(num_kernels):
				if xs is None:
					xs = resblocks[i * num_kernels + j](x)
				else:
					xs += resblocks[i * num_kernels + j](x)
			x = xs / num_kernels
		# x = layers.LeakyReLU()(x) # Original
		x = leakyReLU(x)
		x = conv_post(x)
		# x = tf.math.tanh(x)
		out = tf.math.tanh(x)

		return keras.Model(
			inputs=[inp], outputs=[out], name="Generator"
		)


def get_mpd(input_shape1, input_shape2):
	discriminators = [
		DiscriminatorP(2),
		DiscriminatorP(3),
		DiscriminatorP(5),
		DiscriminatorP(7),
		DiscriminatorP(11),
	]

	y = layers.Input(input_shape1)
	y_hat = layers.Input(input_shape2)
	y_d_rs = []
	y_d_gs = []
	fmap_rs = []
	fmap_gs = []
	for _, d in enumerate(discriminators):
		y_d_r, fmap_r = d(y)
		y_d_g, fmap_g = d(y_hat)
		y_d_rs.append(y_d_r)
		fmap_rs.append(fmap_r)
		y_d_gs.append(y_d_g)
		fmap_gs.append(fmap_g)

	return keras.Model(
		inputs=[y, y_hat], outputs=[y_d_rs, y_d_gs, fmap_rs, fmap_gs],
		name="MultiPeriodDiscriminator"
	)


def get_msd(input_shape1, input_shape2):
	discriminators = [
		DiscriminatorS(use_spectral_norm=True),
		DiscriminatorS(),
		DiscriminatorS(),
	]

	mean_pools = [
		layers.AveragePooling1D(
			pool_size=4, strides=2, padding="same" #padding=2 # valid padding is same or valid
		),
		layers.AveragePooling1D(
			pool_size=4, strides=2, padding="same" #padding=2 # valid padding is same or valid
		),
	]

	y = layers.Input(input_shape1)
	y_hat = layers.Input(input_shape2)
	y_d_rs = []
	y_d_gs = []
	fmap_rs = []
	fmap_gs = []
	for i, d in enumerate(discriminators):
		if i != 0:
			y = mean_pools[i - 1](y)
			y_hat = mean_pools[i - 1](y_hat)
		y_d_r, fmap_r = d(y)
		y_d_g, fmap_g = d(y_hat)
		y_d_rs.append(y_d_r)
		fmap_rs.append(fmap_r)
		y_d_gs.append(y_d_g)
		fmap_gs.append(fmap_g)

	return keras.Model(
		inputs=[y, y_hat], outputs=[y_d_rs, y_d_gs, fmap_rs, fmap_gs],
		name="MultiScaleDiscriminator"
	)
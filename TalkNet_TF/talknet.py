# talknet.py
# 
# Talknet 1/2 implementation in Tensorflow 2.
# Tensorflow 2.4.0
# Python 3.7
# Windows/MacOS/Linux


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from quartznet import BaseBlock


class GaussianEmbedding(layers.Layer):
	def __init__(self, **kwargs):
		super(GaussianEmbedding, self).__init__(**kwargs)

		pass


	def call(self, text, durs):
		pass


class QuartzNet5x5(layers.Model):
	def __init__(self, **kwargs):
		super(QuartzNet5x5, self).__init__(**kwargs)

		# Layers.
		# Padding layer.
		self.pad_1 = layers.ZeroPadding1D(padding=(16, 16))

		# First Conv1D-BatchNorm-Relu block.
		self.sep_conv1 = layers.SeparableConv1D(
			filters=256, kernel_size=33, strides=2, padding="valid",
			data_format="channels_last", depthwise_regularizer=None,
			pointwise_regularizer=None,
		)
		self.batch_norm_1 = layers.BatchNormalization(momentum=0.5)
		self.relu_1 = layers.ReLU()
		self.dropout_1 = layers.Dropout(0.1)

		block_params = [
			[256, 5],
			[256, 7],
			[256, 9],
			[256, 11],
			[256, 13],
		]

		# Next are the sequence of main blocks.
		blocks = []
		for filters, kernel_size in block_params:
			blocks.append(
				BaseBlock(filters, kernel_size, 5)
			)
		self.blocks_model = keras.Sequential(blocks)

		pass


	def call(self, inputs, training=None):
		pass


class QuartzNet9x5(layers.Model):
	def __init__(self, **kwargs):
		super(QuartzNet9x5, self).__init__(**kwargs)

		pass


	def call(self, inputs, training=None):
		pass


class GraphemeDuration(layers.Model):
	def __init__(self, vocab, cfg, **kwargs):
		super(GraphemeDuration, self).__init__(**kwargs)
		self.vocab = vocab
		self.embed = layers.Embedding(len(vocab), cfg.d_char)
		self.encoder = None
		self.proj = layers.Conv1D(1, kernel_size=1)
		pass


	def call(self, text, text_len, is_mask=True):
		embed_out = self.embed(text)
		y, _ = self.encoder(embed_out, length=text_len)
		durs = self.proj(y)
		return durs
		pass


class PitchPredictor(layers.Model):
	def __init__(self, **kwargs):
		super(PitchPredictor, self).__init__(**kwargs)

		pass


	def call(self, text, durs, is_mask=True):
		pass


class SpectrogramModel(layers.Model):
	def __init__(self, vocab, cfg, **kwargs):
		super(SpectrogramModel, self).__init__(**kwargs)
		self.cfg = cfg
		self.vocab = vocab
		self.blanking = cfg.blanking
		self.embed = GaussianEmbedding(self.vocab, cfg.d_char)
		self.norm_f0 = MaskedNorm(1)
		self.res_f0 = StyleResidual(d_char, 1, kernel_size=3)
		self.encoder = cfg.encoder
		self.proj = layers.Conv1D(cfg.n_mels, kernel_size=1)

		pass


	def call(self, text, text_len, durs, f0):
		embed_out, text_len = self.embed(text, durs), durs.sum(-1)
		f0, f0_mask = f0.clone(), f0 > 0.0
		f0 = self.norm_f0(f0, f0_mask)
		f0[~f0_mask] = 0.0
		x = self.res_f0(x, f0)
		y, _ = self.encoder(audio_signal=x, length=text_len)
		mel = self.proj(y)
		return mel
		pass


class TalkNet2(layers.Model):
	def __init__(self, **kwargs):
		super(TalkNet2, self).__init__(**kwargs)

		pass


	def call(self, text, durs, f0, is_mask=True):
		pass
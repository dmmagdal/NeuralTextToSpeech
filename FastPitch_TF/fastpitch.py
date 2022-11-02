# fastpitch.py


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential


class PositionalEmbedding(layers.Layer):
	def __init__(self, emb_dim):
		super(PositionalEmbedding, self).__init__()
		self.emb_dim = emb_dim
		inv_freq = 1 / (10000 ** tf.range(0.0, emb_dim, 2.0) / emb_dim)
		self.inv_freq = inv_freq


	def call(self, pos_seq, bsz=None):
		sinusoid_inp = tf.linalg.matmul(
			tf.expand_dims(pos_seq, -1), 
			tf.expand_dims(self.inv_freq, 0)
		)
		pos_emb = tf.concat(
			[tf.math.sin(sinusoid_inp), tf.math.cos(sinusoid_inp)], 
			axis=1
		)
		if bsz is not None:
			return pos_emb[None, :, :]
		else:
			return pos_emb[None, :, :]


class PositionwiseFF(layers.Layer):
	def __init__(self, model_dim, inner_dim, dropout, pre_lnorm=False):
		super(PositionwiseFF, self).__init__()

		self.model_dim = model_dim
		self.inner_dim = inner_dim
		self.dropout = dropout

		padding = "same" if (kernel_size // 2) else "valid"

		self.core_net = keras.Sequential([
			layers.Dense(inner_dim),
			layers.ReLU(),
			layers.Dropout(dropout),
			layers.Dense(model_dim),
			layers.Dropout(dropout),
		])

		self.layer_norm = layers.LayerNormalization()
		self.pre_lnorm = pre_lnorm


	def call(self, inputs):
		if self.pre_lnorm:
			# Layer normalization + positionwise feed-forward
			core_out = self.core_net(self.layer_norm(inputs))

			# Residual connection
			output = core_out + inputs
		else:
			# Positionwise feed-forward
			core_out = self.core_net(inputs)

			# residual connection + layer normalization
			output = self.layer_norm(inputs + core_out)

		return output


class PositionwiseConvFF(layers.Layer):
	def __init__(self, model_dim, inner_dim, kernel_size, dropout,
			pre_lnorm=False):
		super(PositionwiseConvFF, self).__init__()

		self.model_dim = model_dim
		self.inner_dim = inner_dim
		self.dropout = dropout

		self.core_net = keras.Sequential([
			layers.Conv1D(
				inner_dim, kernel_size=kernel_size, strides=1,
				padding="same" if (kernel_size // 2) != 0 else "causal"#"valid"
			),
			layers.ReLU(),
			layers.Conv1D(
				model_dim, kernel_size=kernel_size, strides=1,
				padding="same" if (kernel_size // 2) != 0 else "causal"#"valid"
			),
			layers.Dropout(dropout),
		])

		self.layer_norm = layers.LayerNormalization()
		self.pre_lnorm = pre_lnorm


	def call(self, inputs):
		if self.pre_lnorm:
			# Layer normalization + positionwise feed-forward
			core_out = self.core_net(self.layer_norm(inputs))

			# Residual connection
			output = core_out + inputs
		else:
			# Positionwise feed-forward
			core_out = self.core_net(inputs)

			# residual connection + layer normalization
			output = self.layer_norm(inputs + core_out)

		return output


class TransformerLayer(layers.Layer):
	def __init__(self, n_head, model_dim, head_dim, inner_dim, 
			kernel_size, dropout, **kwargs):
		super(TransformerLayer, self).__init__()

		self.dec_attn = layers.MultiHeadAttention(
			n_head, model_dim, dropout, **kwargs
		)
		self.pos_ff = PositionwiseConvFF(
			model_dim, inner_dim, kernel_size, dropout,
			pre_lnorm=kwargs.get('pre_lnorm')
		)


	def call(self, dec_inputs, mask=None):
		output = self.dec_attn(dec_inputs, attn_mask=~tf.squeeze(mask, 1))
		output *= mask
		output = self.pos_ff(output)
		output *= mask
		return output


class Transformer(layers.Layer):
	def __init__(self, n_layer, n_head, d_model, d_head, d_inner,
			kernel_size, dropout, dropatt, dropemb=0.0, 
			embed_input=True, n_embed=None, d_embed=None, 
			pre_lnorm=False):
		super(Transformer, self).__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.d_head = d_head

		if embed_input:
			self.word_emb = layers.Embedding(
				n_embed, d_embed or d_model
			)
		else:
			self.word_emb = None

		self.pos_emb = PositionalEmbedding(self.d_model)
		self.drop = layers.Dropout(dropemb)
		self.layers = Sequential([
			TransformerLayer(
				n_head, d_model, d_head, d_inner, kernel_size,
				dropout, dropatt=dropatt, pre_lnorm=pre_lnorm
			) for _ in range(n_layer)
		])


	def call(self, dec_inp, seq_lens=None, conditioning=0):
		if self.word_emb is None:
			inp = dec_inp
			mask = tf.expand_dims(mask_from_lens(seq_lens), 2)
		else:
			inp = self.word_emb(dec_inp)
			mask = tf.expand_dims((dec_inp != 0), 2)

		pos_seq = tf.range(inp.shape[1])
		pos_emb = self.pos_emb(pos_seq) * mask

		out = self.drop(inp + pos_emb + conditioning)

		for layer in self.layers:
			out = layer(out, mask=mask)

		return out, mask
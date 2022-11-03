# transformer.py


from typing import List, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from common.utils import mask_from_lens


class PositionalEmbedding(layers.Layer):
	def __init__(self, emb_dim):
		super(PositionalEmbedding, self).__init__()
		self.emb_dim = emb_dim
		inv_freq = 1 / (10000 ** tf.range(0.0, emb_dim, 2.0) / emb_dim)
		self.inv_freq = inv_freq


	def call(self, pos_seq, bsz=None):
		sinusoid_inp = tf.linalg.matmul(
			tf.expand_dims(pos_seq, -1), tf.expand_dims(self.inv_freq, 0)
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


class MultiHeadAttn(layers.Layer):
	def __init__(self, n_head, model_dim, head_dim, dropout, 
			dropatt=0.1, pre_lnorm=False):
		super(MultiHeadAttn, self).__init__()

		self.n_head = n_head
		self.model_dim = model_dim
		self.head_dim = head_dim
		self.scale = 1 / (head_dim ** 0.5)
		self.dropout = dropout
		self.pre_lnorm = pre_lnorm

		self.qkv_net = layers.Dense(3 * n_head * head_dim)
		self.drop = layers.Dropout(dropout)
		self.drop_att = layers.Dropout(dropatt)
		self.o_net = layers.Dense(model_dim, use_bias=False)
		self.layer_norm = layers.LayerNormalization()


	def call(self, inputs, attn_mask: Optional[tf.Tensor]=None):
		residual = inputs

		if self.pre_lnorm:
			# layer normalization
			inputs = self.layer_norm(inputs)

		n_head, head_dim = self.n_head, self.head_dim

		# head_q, head_k, head_v = 


class TransformerLayer(layers.Layer):
	def __init__(self, n_head, model_dim, head_dim, inner_dim, 
			kernel_size, dropout, **kwargs):
		super(TransformerLayer, self).__init__()

		self.dec_attn = MultiHeadAttn(
			n_head, model_dim, head_dim, dropout, **kwargs
		)
		self.pos_ff = PositionwiseConvFF(
			model_dim, inner_dim, kernel_size, dropout,
			pre_lnorm=kwargs.get('pre_lnorm')
		)


	def call(self, dec_inputs, mask):
		output = self.dec_attn(dec_inputs, attn_mask=~mask)
		output *= mask
		output = self.pos_ff(output)
		output *= mask
		return output


class FFTransformer(layers.Layer):
	def __init__(self, n_layer, n_head, model_dim, head_dim, 
			inner_dim, kernel_size, dropout, dropatt, dropemb=0.0,
			embed_input=True, n_emb=None, emb_dim=None,
			padding_idx=0, pre_lnorm=False):
		super(FFTransformer, self).__init__()

		self.model_dim = model_dim
		self.n_head = n_head
		self.head_dim = head_dim
		self.padding_idx = padding_idx
		self.n_emb = n_emb

		self.embed_input = embed_input
		if self.embed_input:
			self.word_emb = layers.Embedding(n_emb, emb_dim or model_dim)
		else:
			self.word_emb = None

		self.pos_emb = PositionalEmbedding(self.model_dim)
		self.drop = layers.Dropout(dropemb)
		self.layers = []

		for _ in range(n_layer):
			self.layers.append(
				TransformerLayer(
					n_head, model_dim, head_dim, inner_dim, 
					kernel_size, dropout, dropatt=dropatt,
					pre_lnorm=pre_lnorm
				)
			)


	def call(self, dec_inputs, seq_lens: Optional[tf.Tensor]=None,
			conditioning: Optional[tf.Tensor]=None):
		if not self.embed_input:
			inputs = dec_inputs
			assert seq_lens is not None
			mask = mask_from_lens(seq_lens)
		else:
			inputs = self.word_emb(dec_inputs)
			mask = (dec_inputs != self.padding_idx)

		pos_seq = tf.range(inputs.shape[1])
		pos_emb = self.pos_emb(pos_seq) * mask
		if conditioning is not None:
			out = self.drop(inputs + pos_emb + conditioning)
		else:
			out = self.drop(inputs + pos_emb)

		for layer in self.layers:
			out = layer(out, mask=mask)

		return out, mask
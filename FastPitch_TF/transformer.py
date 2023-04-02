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
		self.inv_freq = tf.Variable(inv_freq, trainable=False)


	def call(self, pos_seq, bsz=None):
		sinusoid_inp = tf.linalg.matmul(
			tf.expand_dims(pos_seq, -1), tf.expand_dims(self.inv_freq, 0)
		)
		pos_emb = tf.concat(
			[tf.math.sin(sinusoid_inp), tf.math.cos(sinusoid_inp)], 
			axis=-1
		)
		if bsz is not None:
			return tf.broadcast_to(pos_emb[None, :, :], [bsz, -1, -1])
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
				padding="same" if (kernel_size // 2) != 0 else "valid"#"causal"#"valid"
			),
			layers.ReLU(),
			layers.Conv1D(
				model_dim, kernel_size=kernel_size, strides=1,
				padding="same" if (kernel_size // 2) != 0 else "valid"#"causal"#"valid"
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

		head_q, head_k, head_v = tf.split(
			self.qkv_net(inputs), 3, axis=-1
		)
		# head_q = head_q.reshape(
		# 	inputs.shape[0], inputs.shape[1], n_head, d_head
		# )
		# head_k = head_k.reshape(
		# 	inputs.shape[0], inputs.shape[1], n_head, d_head
		# )
		# head_v = head_v.reshape(
		# 	inputs.shape[0], inputs.shape[1], n_head, d_head
		# )
		head_q = tf.reshape(
			head_q,
			[inputs.shape[0], inputs.shape[1], n_head, head_dim]
		)
		head_k = tf.reshape(
			head_k,
			[inputs.shape[0], inputs.shape[1], n_head, head_dim]
		)
		head_v = tf.reshape(
			head_v,
			[inputs.shape[0], inputs.shape[1], n_head, head_dim]
		)

		# q = tf.transpose(head_q, [2, 0, 1, 3]).reshape(
		# 	[-1, inputs.shape[1], d_head]
		# )
		# k = tf.transpose(head_k, [2, 0, 1, 3]).reshape(
		# 	[-1, inputs.shape[1], d_head]
		# )
		# v = tf.transpose(head_v, [2, 0, 1, 3]).reshape(
		# 	[-1, inputs.shape[1], d_head]
		# )
		q = tf.reshape(
			tf.transpose(head_q, [2, 0, 1, 3]),
			[-1, inputs.shape[1], head_dim]
		)
		k = tf.reshape(
			tf.transpose(head_k, [2, 0, 1, 3]),
			[-1, inputs.shape[1], head_dim]
		)
		v = tf.reshape(
			tf.transpose(head_v, [2, 0, 1, 3]),
			[-1, inputs.shape[1], head_dim]
		)

		attn_score = tf.linalg.matmul(q, tf.transpose(k, [0, 2, 1]))
		attn_score = attn_score * self.scale

		if attn_mask is not None:
			attn_mask = tf.cast(
				tf.expand_dims(attn_mask, 1), dtype=attn_score.dtype
			)
			# attn_mask = tf.repeat(
			attn_mask = tf.tile(
				attn_mask, [n_head, attn_mask.shape[-1], 1]
			)
			attn_score = tf.where(
				tf.cast(attn_mask, tf.bool), attn_score, 
				tf.fill(attn_score.shape, -float('inf'))
			)

		attn_prob = tf.nn.softmax(attn_score, axis=-1)
		attn_prob = self.drop_att(attn_prob)
		attn_vec = tf.linalg.matmul(attn_prob, v)

		# attn_vec = tf.reshape(
		# 	attn_vec, [n_head, inputs.shape[0], inputs.shape[1], d_head]
		# )
		# attn_vec = tf.reshape(
		# 	tf.transpose(attn_mask, [1, 2, 0, 3]), 
		# 	[inputs.shape[0], inputs.shape[1], n_head * d_head]
		# )
		attn_vec = tf.reshape(
			attn_vec, [n_head, inputs.shape[0], inputs.shape[1], head_dim]
		)
		attn_vec = tf.reshape(
			tf.transpose(attn_vec, perm=[1, 2, 0, 3]), 
			[inputs.shape[0], inputs.shape[1], n_head * head_dim]
		)

		# Linear projection
		attn_out = self.o_net(attn_vec)
		attn_out = self.drop(attn_out)

		if self.pre_lnorm:
			# Residual connection
			output = residual + attn_out
		else:
			# Residual connection + layer normalization
			output = self.layer_norm(residual + attn_out)

		output = tf.cast(output, dtype=attn_out.dtype)

		return output


class TransformerLayer(layers.Layer):
	def __init__(self, n_head, model_dim, head_dim, inner_dim, 
			kernel_size, dropout, **kwargs):
		super(TransformerLayer, self).__init__()

		self.dec_attn = MultiHeadAttn(
			n_head, model_dim, head_dim, dropout, **kwargs
		)
		# self.dec_attn = layers.MultiHeadAttention(
		# 	n_head, model_dim, dropout=dropout
		# )
		self.pos_ff = PositionwiseConvFF(
			model_dim, inner_dim, kernel_size, dropout,
			pre_lnorm=kwargs.get('pre_lnorm')
		)


	def call(self, dec_inputs, mask=None):
		x = ~mask
		x = tf.squeeze(~mask, axis=-1) # Used -1 instead of 2 because mask is not (b, len, dim) only (len, dim)
		output = self.dec_attn(
			# dec_inputs, attn_mask=tf.squeeze(~mask, axis=2)
			dec_inputs, attn_mask=tf.squeeze(~mask, axis=-1)
		)
		# output = self.dec_attn(
		# 	dec_inputs, 
		# 	dec_inputs,
		# 	attention_mask=~mask#tf.squeeze(~mask, axis=-1)
		# )
		output *= tf.cast(mask, dtype=tf.float32) # Cast mask to float because tensorflow cannot do the implicit type conversion from bool to float like pytorch
		output = self.pos_ff(output)
		output *= tf.cast(mask, dtype=tf.float32) # Again, cast to float
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
			# mask [batch_size, seq_len, 1]
			mask = (dec_inputs != self.padding_idx)
		mask = tf.expand_dims(mask, -1)
		pos_seq = tf.range(inputs.shape[1], dtype=inputs.dtype)

		pos_seq = tf.range(inputs.shape[1], dtype=inputs.dtype) # inputs.shape[1] = len(encoded_input_text)
		pos_emb = self.pos_emb(pos_seq) * tf.cast(mask, tf.float32)
		if conditioning is not None:
			out = self.drop(inputs + pos_emb + conditioning)
		else:
			out = self.drop(inputs + pos_emb)

		for layer in self.layers:
			out = layer(out, mask=mask)

		return out, mask
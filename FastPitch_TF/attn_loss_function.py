# attn_loss_function.py


import tensorflow as tf
from tensorflow import keras


class AttentionCTCLoss:
	def __init__(self, blank_logprob=-1):
		# self.log_softmax = tf.nn.log_softmax()
		self.blank_logprob = blank_logprob
		# self.ctc_loss = tf.nn.ctc_loss()


	def call(self, attn_logprob, in_lens, out_lens):
		key_lens = in_lens
		query_lens = out_lens
		pad = [[0, 0], [0, 0], [0, 0], [1, 0]]
		attn_logprob_padded = tf.pad(
			attn_logprob, pad, constant_values=self.blank_logprob
		)
		cost_total = 0.0
		for bid in range(attn_logprob.shape[0]):
			target_seq = tf.expand_dims(
				tf.range(1, key_lens[bid] + 1), axis=0
			)
			curr_logprob = tf.transpose(
				attn_logprob_padded[bid], [1, 0, 2]
			)
			curr_logprob = curr_logprob[
				:query_lens[bid], :, :key_lens[bid]
			]
			# curr_logprob = self.log_softmax(curr_logprob[None])[0]
			curr_logprob = tf.nn.log_softmax(curr_logprob[None])[0]
			# input_labels, target_labels, input_len, logit_len
			# ctc_cost = self.ctc_cost(
			ctc_cost = tf.nn.ctc_loss(
				target_seq, curr_logprob, query_lens[bid:bid + 1],
				key_lens[bid:bid + 1]
			)
			ctc_total += ctc_cost
		cost = ctc_total / attn_logprob.shape[0]
		return cost


class AttentionBinarizationLoss:
	def __init__(self):
		pass


	def call(self, hard_attention, soft_attention, eps=1e-12):
		log_sum = tf.math.reduce_sum(
			tf.math.log(
				tf.clip_by_value(
					soft_attention[hard_attention == 1], 
					clip_value_min=eps, clip_value_max=tf.float32.max
				)
			)
		)
		return -log_sum / tf.math.reduce_sum(hard_attention)
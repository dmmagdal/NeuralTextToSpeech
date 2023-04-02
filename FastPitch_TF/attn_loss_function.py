# attn_loss_function.py


import tensorflow as tf
from tensorflow import keras


class AttentionCTCLoss:
# class AttentionCTCLoss(keras.losses.Loss):
	def __init__(self, blank_logprob=-1):
		# super(AttentionCTCLoss, self).__init__()
		# self.log_softmax = tf.nn.log_softmax()
		self.blank_logprob = blank_logprob
		# self.ctc_loss = tf.nn.ctc_loss()


	def __call__(self, attn_logprob, in_lens, out_lens):
	# def call(self, attn_logprob, in_lens, out_lens):
		key_lens = in_lens
		query_lens = out_lens
		max_key_len = attn_logprob.shape[-1]

		# Reorder input to [query_len, batch_size, key_len].
		attn_logprob = tf.squeeze(attn_logprob, 1)
		attn_logprob = tf.transpose(attn_logprob, [1, 0, 2])

		# Add blank label.
		# pad = [[0, 0], [0, 0], [0, 0], [1, 0]]
		pad = [[0, 0], [0, 0], [1, 0]]
		attn_logprob = tf.pad(
			attn_logprob, pad, constant_values=self.blank_logprob
		)

		# Convert to log probabilities. Note: mask out probs beyond
		# key_len.
		key_inds = tf.range(max_key_len + 1, dtype=tf.int64)
		mask = key_inds[None, None, :] > key_lens[None, :, None]
		attn_logprob = tf.where(mask, -float("inf"), attn_logprob)
		attn_logprob = tf.nn.softmax(attn_logprob, axis=-1)

		# Target sequences.
		target_seqs = tf.expand_dims(key_inds[1:], 0)
		target_seqs = tf.tile(target_seqs, [tf.shape(key_lens)[0], 1])

		# Evaluate CTC loss.
		cost = tf.nn.ctc_loss(
			target_seqs, attn_logprob, key_lens, query_lens
		)

		'''
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
		'''
		return cost


class AttentionBinarizationLoss(keras.losses.Loss):
	def __init__(self):
		super(AttentionBinarizationLoss, self).__init__()


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
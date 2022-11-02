# utils.py


from collections import defaultdict, OrderedDict
import tensorflow as tf


def mask_from_lens(lengths):
	max_len = tf.math.reduce_max(lengths).numpy().item()
	mask = tf.sequence_mask(lengths, max_len, dtype=tf.bool)
	return mask


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


class DefaultAttrDict(defaultdict):
	def __init__(self, *args, **kwargs):
		super(DefaultAttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

	def __getattr__(self, item):
		return self[item]
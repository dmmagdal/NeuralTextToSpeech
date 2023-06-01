# params.py


import numpy as np


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


	def override(self, attrs):
		if isinstance(attrs, dict):
			self.__dict__.update(**attrs)
		elif isinstance(attrs, (list, tuple, set)):
			for attr in attrs:
				self.override(attr)
		elif attrs is not None:
			raise NotImplementedError
		return self


params = AttrDict(
	# Training params
	batch_size = 4,#16,
	learning_rate = 2e-4,
	max_grad_norm = None,

	# Data params
	sample_rate = 22050,
	n_mels = 80,
	n_fft = 1024,
	hop_length = 256,		# Renamed myself from hop_samples
	win_length = 1024,		# Added myself
	crop_mel_frames = 62,	# Probably an error in paper. Original
	f_min = 0,				# Added myself
	f_max = 8000,			# Added myself

	# Model params
	residual_layers = 30,
	residual_channels = 64,
	dilation_cycle_length = 10,
	unconditional = False,
	noise_schedule = np.linspace(1e-4, 0.05, 50).tolist(),
	inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

	# unconditional sample len
	audio_len = 22050 * 5, # unconditional_synthesis_samples
)
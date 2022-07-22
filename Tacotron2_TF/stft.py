# stft.py


import numpy as np
import tensorflow as tf
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from audio_processing import window_sumsquare


class STFT:
	def __init__(self, filter_length=800, hop_length=200, 
			win_length=800, window='hann'):
		# super(STFT, self).__init__()
		self.filter_length = filter_length
		self.hop_length = hop_length
		self.win_length = win_length
		self.window = window
		self.forward_transform = None
		scale = self.filter_length / self.hop_length
		fourier_basis = np.fft.fft(np.eye(self.filter_length))

		cutoff = int((self.filter_length / 2 + 1))
		fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
								   np.imag(fourier_basis[:cutoff, :])])

		forward_basis = tf.convert_to_tensor(
			fourier_basis[:, None, :], dtype=tf.float32
		)
		inverse_basis = tf.convert_to_tensor(
			np.linalg.pinv(scale * fourier_basis).T[:, None, :], 
			dtype=tf.float32
		)

		if window is not None:
			assert(filter_length >= win_length)
			# get window and zero center pad it to filter_length
			fft_window = get_window(window, win_length, fftbins=True)
			fft_window = pad_center(fft_window, filter_length)
			# fft_window = torch.from_numpy(fft_window).float()
			fft_window = tf.convert_to_tensor(
				fft_window, dtype=tf.float32
			)

			# window the bases
			forward_basis *= fft_window
			inverse_basis *= fft_window

		# self.register_buffer('forward_basis', forward_basis.float())
		# self.register_buffer('inverse_basis', inverse_basis.float())
		self.forward_basis = tf.convert_to_tensor(
			forward_basis, dtype=tf.float32
		)
		self.inverse_basis = tf.convert_to_tensor(
			inverse_basis, dtype=tf.float32
		)


	def transform(self, input_data):
		# num_batches = input_data.size(0)
		# num_samples = input_data.size(1)
		num_batches = input_data.shape[0]
		num_samples = input_data.shape[1]

		self.num_samples = num_samples

		# similar to librosa, reflect-pad the input
		# input_data = input_data.view(num_batches, 1, num_samples)
		input_data = tf.reshape(
			input_data, [num_batches, 1, num_samples]
		)
		input_data = F.pad(
			input_data.unsqueeze(1),
			(int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
			mode='reflect')
		input_data = input_data.squeeze(1)

		forward_transform = F.conv1d(
			input_data,
			Variable(self.forward_basis, requires_grad=False),
			stride=self.hop_length,
			padding=0)

		cutoff = int((self.filter_length / 2) + 1)
		real_part = forward_transform[:, :cutoff, :]
		imag_part = forward_transform[:, cutoff:, :]

		magnitude = torch.sqrt(real_part**2 + imag_part**2)
		phase = torch.autograd.Variable(
			torch.atan2(imag_part.data, real_part.data))

		return magnitude, phase

	def inverse(self, magnitude, phase):
		recombine_magnitude_phase = torch.cat(
			[magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

		inverse_transform = F.conv_transpose1d(
			recombine_magnitude_phase,
			Variable(self.inverse_basis, requires_grad=False),
			stride=self.hop_length,
			padding=0)

		if self.window is not None:
			window_sum = window_sumsquare(
				self.window, magnitude.shape[-1], hop_length=self.hop_length,
				win_length=self.win_length, n_fft=self.filter_length,
				dtype=np.float32)
			# remove modulation effects
			approx_nonzero_indices = tf.convert_to_tensor(
				np.where(window_sum > tiny(window_sum))[0]
			)
			# approx_nonzero_indices = torch.from_numpy(
			# 	np.where(window_sum > tiny(window_sum))[0])
			window_sum = tf.convert_to_tensor(window_sum)
			# window_sum = torch.autograd.Variable(
			# 	torch.from_numpy(window_sum), requires_grad=False)
			# window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
			inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

			# scale by hop ratio
			inverse_transform *= float(self.filter_length) / self.hop_length

		inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
		inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

		return inverse_transform

	def forward(self, input_data):
		self.magnitude, self.phase = self.transform(input_data)
		reconstruction = self.inverse(self.magnitude, self.phase)
		return reconstruction
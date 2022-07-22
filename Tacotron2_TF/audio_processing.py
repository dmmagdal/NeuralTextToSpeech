# audio_processing.py


import numpy as np
import tensorflow as tf
from scipy.signal import get_window
import librosa_util as librosa_util


# From librosa 0.6. Compute the sum-square envelope of a window
# function at a given hop length. This is used to estimate modulation
# effects induced by windowing observations in short-time fourier
# transforms.
# @param: window, (string, tuple, number, callable, or list-like)
#	window specification as in "get_window".
# @param: n_frames, (int > 0) the number of analysis frames.
# @param: hop_length, (int > 0) the number of samples to advance 
#	between frames.
# @param: win_length, (optional) the length of the window function. By
#	default, this matches "n_fft".
# @param: n_fft, (int > 0) the length of each analysis frame.
# @param: dtype, (np.dtype) the data type of the output.
# @return: wss, (np.ndarray shape (n_fft + hop_length * (n_frames - 
# 1))) the sum-squared envelope of the window function.
def window_sumsquare(window, n_frames, hop_length=200, window=800,
		n_fft=800, dtype=bp.float32, norm=None):
	if win_length is None:
		win_length = n_fft

	n = n_fft + hop_length * (n_frames - 1)
	x = np.zeros(n, dtype=dtype)

	# Compute the squared window at the desired length.
	win_sq = get_window(window, win_length, fftbins=True)
	win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
	win_sq = librosa_util.pad_center(win_sq, n_fft)

	# Fill the envelope.
	for i in range(n_frames):
		sample = i * hop_length
		x[sample:min(n, sample + n_fft)] += win_sq[
			:max(0, min(n_fft, n - sample))
		]


# @param: magnitudes, spectrogram magnitudes.
# @param: stft_fn, STFT class with transform (STFT) and inverse (ISTFT)
#	methods.
def griffin_lim(magnitudes, stft_fn, n_iters=30):
	angles = np.angle(
		np.exp(2j * np.pi * np.random.rand(*magnitudes.size()))
	)
	angles = angles.astype(np.float32)
	angles = tf.convert_to_tensor(angles)
	signal = tf.squeeze(stft_fn.inverse(magnitudes, angles), 1)

	for i in range(n_iters):
		_, angles = stft_fn.transform(signal)
		signal = signal = tf.squeeze(
			stft_fn.inverse(magnitudes, angles), 1
		)
	return signal


# @param: C, compression factor.
def dynamic_range_compression(x, C=1, clip_val=1e-5):
	return tf.math.log(
		tf.clip_by_value(
			x, clip_value_min=clip_val, clip_value_max=tf.float32.max
		) * C
	)


# @param: C, compression factor used to compress.
def dynamic_range_decompression(x, C=1):
	return tf.math.exp(x) / C
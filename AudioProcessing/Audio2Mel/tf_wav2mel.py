# tf_wav2mel.py
# Reference: https://www.tensorflow.org/api_docs/python/tf/signal/hann_window
# Reference: https://www.tensorflow.org/api_docs/python/tf/signal/stft
# Reference: https://www.tensorflow.org/api_docs/python/tf/pad


import tensorflow as tf
from librosa.filters import mel as librosa_mel_fn


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, 
		win_size, fmin, fmax, center=False):
	if tf.math.reduce_min(y) < -1.0:
		print("min value is ", tf.math.reduce_min(y))
	if tf.math.reduce_max(y) > 1.0:
		print("max value is ", tf.math.reduce_max(y))

	mel_basis, hann_window = {}, {}
	if fmax not in mel_basis:
		mel = librosa_mel_fn(
			sampling_rate, n_fft, num_mels, fmin, fmax
		)
		mel_basis[str(fmax) + "_cpu"] = tf.convert_to_tensor(mel, dtype=tf.float32)
		hann_window["cpu"] = tf.signal.hann_window(win_size)

	y = tf.pad(
		tf.expand_dims(y, 1), 
		(int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
		mode='reflect'
	)
	y = tf.squeeze(y, 1)

	spec = tf.signal.stft(
		y, frame_length=n_fft, frame_step=hop_size, 
		fft_length=win_size, window_fn=hann_window["cpu"], #center=center,
		#pad_mode="reflect", normalized=False, onesided=True
	)
	spec = tf.math.sqrt(
		tf.math.reduce_sum(tf.math.pow(spec, 2), -1) + (1e-9)
	)

	spec = tf.linalg.matmul(
		mel_basis[str(fmax) + "_cpu"], spec
	)
	spec = spectral_normalize_tf(spec)

	return spec


def spectral_normalize_tf(magnitudes):
	output = dynamic_range_compression_tf(magnitudes)
	return output


def dynamic_range_compression_tf(x, C=1, clip_val=1e-5):
	return tf.math.log(
		tf.clip_by_value(x, min=clip_val, max=tf.float32.max) * C
	)
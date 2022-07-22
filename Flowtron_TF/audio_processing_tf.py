# audio_processing.py
# Apply audio processing to wav files. Should be similar to the
# Tacotron 2 STFT function from the Nvidia Tacotron 2 repo
# (https://github.com/NVIDIA/tacotron2).
# Source (TF Documentation STFT): https://www.tensorflow.org/api_docs/
#	python/tf/signal/stft
# Source (TF Documentation Linear to Mel Weight Matrix): 
#	https://www.tensorflow.org/api_docs/python/tf/signal/
#	linear_to_mel_weight_matrix
# Source (Keras Example MelGAN): https://keras.io/examples/audio/
#	melgan_spectrogram_inversion/
# Source (Medium): https://towardsdatascience.com/how-to-easily-
#	process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
# Source (Keras Example Transformer ASR): https://keras.io/examples/
#	audio/transformer_asr/
# Source (Keras Example ASR with CTC): https://keras.io/examples/audio/
#	ctc_asr/
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf


def dynamic_range_compression(x, C=1, clip_val=1e-5):
	return tf.math.log(tf.clip_by_value(
		x, clip_value_min=clip_val, clip_value_max=tf.float32.max) * C
   )


class STFT:
	# Used the audio processing in the Keras MelGAN example
	# (https://keras.io/examples/audio/melgan_spectrogram_inversion/)
	# and Medium article (https://towardsdatascience.com/how-to-easily-
	# process-audio-on-your-gpu-with-tensorflow-2d9d91360f06) to
	# process the audio into mel spectrograms with Tensorflow.
	def __init__(self, filter_length=1024, frame_length=1024, 
			frame_step=256,	n_mel_channels=80, sampling_rate=22050, 
			mel_fmin=0.0, mel_fmax=8000.0):
		self.frame_length = frame_length
		self.frame_step = frame_step
		self.fft_length = filter_length
		self.sampling_rate = sampling_rate
		self.n_mel_channels = n_mel_channels
		self.mel_fmin = mel_fmin
		self.mel_fmax = mel_fmax

		# Mel filter blank (to be multiplied with STFT output).
		self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
			num_mel_bins=self.n_mel_channels,
			num_spectrogram_bins=self.frame_length // 2 + 1,
			sample_rate=self.sampling_rate,
			lower_edge_hertz=self.mel_fmin, 
			upper_edge_hertz=self.mel_fmax
		)
		

	# Convert a wav signal to mel spectrogram.
	# @param: x, TF.Tensor (of shape (wav_length,) or (wav_length, 1))
	#	that is the input signal from the wav file.
	# @return: returns a TF.Tensor of shape 
	#	(some_length, n_mel_channels) that is the mel spectrogram of
	#	the input signal.
	def mel_spectrogram(self, x):
		# Adjust the shape of the signal if it is of shape
		# (wav_length, 1).
		if len(tf.shape(x)) == 2:
			# x = tf.squeeze(x, axis=-1)
			x = tf.squeeze(x, axis=0)

		assert len(tf.shape(x)) == 1, "Invalid signal shape {}, expected (wav_length,)".format(tf.shape(x))

		# Convert signal to spectrogram with STFT function.
		spectrogram = tf.signal.stft(
			x, frame_length=self.frame_length, 
			frame_step=self.frame_step, fft_length=self.fft_length
		)

		# Get magnitude of spectrogram.
		magnitude = tf.abs(spectrogram)

		# Multiply the mel filterbank with the magnitude.
		mel_spec = tf.linalg.matmul(
			tf.math.square(magnitude), self.mel_filterbank
		)

		# Apply spectral normalization through the dynamic range
		# compression normalization.
		mel_spec = dynamic_range_compression(mel_spec)

		# Return the mel spectrogram.
		return mel_spec

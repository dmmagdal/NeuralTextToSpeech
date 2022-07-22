# process_audio.py
# Take the wav filepath and extract the mel-spectrogram.


import librosa
import tensorflow as tf
import tensorflow_io as tfio


#######################################################################
# Tensorflow
# Source: https://keras.io/examples/audio/ctc_asr/#preprocessing
#######################################################################
# Use tensorflow to extract mel spectrogram from wav file.
def get_mel_tf(filepath, frame_length, frame_step, fft_length):
	# Note: frame_length = win_length, frame_step = n_frames,
	# fft_length = n_fft.

	# 1. Read wav file
	file = tf.io.read_file(filepath)

	# 2. Decode the wav file
	audio, _ = tf.audio.decode_wav(file)
	audio = tf.squeeze(audio, axis=-1)

	# 3. Change type to float
	audio = tf.cast(audio, tf.float32)

	# 4. Get the spectrogram
	spectrogram = tf.signal.stft(
		audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
	)

	# 5. We only need the magnitude, which can be derived by applying
	# tf.abs
	spectrogram = tf.abs(spectrogram)
	spectrogram = tf.math.pow(spectrogram, 0.5)
	
	# 6. normalisation
	means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
	stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
	spectrogram = (spectrogram - means) / (stddevs + 1e-10)
	return spectrogram


#######################################################################
# Tensorflow IO
# Source: https://www.tensorflow.org/io/tutorials/audio
#######################################################################
def get_mel_tfio(filepath):
	# Read in audio tensor.
	audio = tfio.audio.AudioTensor(filepath)

	# Remove last dimension. Optional: trim noise and fade in/out.
	audio_tensor = tf.squeeze(audio, axis=-1)

	# Convert to spectrogram.
	spectrogram = tfio,audio.spectrogram(
		audio_tensor, nfft=n_fft, window=win_length, stride=frame_step
	)

	# Convert to mel-spectrogram.
	mel_spectrogram = tfio.audio.melscale(
		spectrogram, rate=sample_rate, mels=n_mels, fmin=0, fmax=8000
	)
	return mel_spectrogram


#######################################################################
# Librosa 
# Source: https://stackoverflow.com/questions/69387104/how-to-convert-
#	wav-audio-file-from-mel-spectrogram
#######################################################################
# Use librosa library to extract mel spectrogram from wav file.
def get_mel_librosa(filepath, n_fft=2048, hop_length=512, 
		win_length=None, n_mels=128):
	audio, sample_rate = librosa.load(filepath)
	mel_spec = librosa.feature.melspectrogram(
		y=audio,
		sr=sample_rate,
		n_fft=n_fft,
		hop_length=hop_length,
		win_length=win_length,
		window="hann",
		center=True,
		pad_mode="reflect",
		power=2.0,
		n_mels=n_mels
	)
	return mel_spec


# Use librosa library to convert mel spectrogram to wav file.
def get_wav_librosa(filepath, mel_spec, sample_rate, n_fft=2048, 
		hop_length=512, win_length=None, n_iter=32):
	audio = librosa.feature.inverse.mel_to_audio(
		y=mel_spec,
		sr=sample_rate,
		n_fft=n_fft,
		hop_length=hop_length,
		win_length=win_length,
		window="hann",
		center=True,
		pad_mode="reflect",
		power=2.0,
		n_iter=n_iter
	)
	return audio
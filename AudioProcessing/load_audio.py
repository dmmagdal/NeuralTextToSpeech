# load_audio.py

import torchaudio
import tensorflow as tf
import librosa
from scipy.io.wavfile import read


def load_wav_scipy(full_path):
	# Read with scipy.
	sampling_rate, audio = read(full_path)
	
	return audio, sampling_rate


def load_wav_tf(full_path):
	# Read with tensorflow.
	file = tf.io.read_file(full_path)
	audio, sampling_rate = tf.audio.decode_wav(file)
	
	# Extra step required to reduce the dims.
	audio = tf.squeeze(audio, axis=-1)
	audio = tf.cast(audio, tf.float32)

	return audio, sampling_rate


def load_wav_librosa(full_path):
	# Read with librosa.
	audio, sampling_rate = librosa.load(full_path)

	return audio, sampling_rate


def load_wav_torchaudio(full_path):
	# Read with torchaudio.
	audio, sampling_rate = torchaudio.load(full_path)

	# Extra step required to reduce the dims.
	audio = audio.squeeze()

	return audio, sampling_rate
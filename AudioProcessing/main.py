# main.py
# Run through the different modules and ways to read (wav) audio files
# and convert the data to mel-spectrograms the same way done in librosa
# or the PyTorch Tacotron STFT module.


import numpy as np
import tensorflow as tf
from load_audio import load_wav_scipy, load_wav_librosa, load_wav_tf


def main():
	# Hyperparameters & variables.
	path = "./LJ001-0001.wav"

	# -----------------------------------------------------------------
	# Test the different ways of loading a wav file.
	print("READ AUDIO FILE:")

	# Scipy.
	scipy_audio, scipy_sr = load_wav_scipy(path)
	print(f"Scipy read shape: {scipy_audio.shape}")

	# Librosa.
	librosa_audio, librosa_sr = load_wav_librosa(path)
	print(f"Librosa read shape: {librosa_audio.shape}")

	# Tensorflow.
	tf_audio, tf_sr = load_wav_tf(path)
	print(f"Tensorflow read shape: {tf_audio.shape}")

	# All data is read to the same shape.
	shape_match1 = scipy_audio.shape == librosa_audio.shape
	shape_match2 = scipy_audio.shape == tf_audio.shape

	# All sampling rates are the same.
	sr_match1 = scipy_sr == librosa_sr
	sr_match2 = scipy_sr == tf_sr

	# All data arrays (converted to numpy) are the same.
	data_match1 = np.array_equal(scipy_audio, librosa_audio)
	data_match2 = np.array_equal(librosa_audio, tf_audio.numpy())
	data_match3 = np.array_equal(scipy_audio, tf_audio.numpy())

	# Output comparisons.
	print(f"All audio shapes match: {shape_match1 and shape_match2}")
	print(f"All sampling rates match: {sr_match1 and sr_match2}")
	print(f"Data matching (raw):")
	print(f"\tLibrosa ({librosa_audio.dtype}) <-> Scipy ({scipy_audio.dtype}): {data_match1}")
	print(f"\tLibrosa ({librosa_audio.dtype}) <-> Tensorflow ({tf_audio.numpy().dtype}): {data_match2}")
	print(f"\tScipy ({scipy_audio.dtype}) <-> Tensorflow ({tf_audio.numpy().dtype}): {data_match3}")

	# Convert all audio data (except the audio from scipy) into numpy 
	# int16 and compare the data.
	librosa_int16 = librosa_audio.astype(np.int16)
	tf_int16 = tf_audio.numpy().astype(np.int16)
	int16_match1 = np.array_equal(scipy_audio, librosa_int16)
	int16_match2 = np.array_equal(librosa_int16, tf_int16)
	int16_match3 = np.array_equal(scipy_audio, tf_int16)

	# Output these new comparisons. 
	print(f"Data matching (int16 conversion):")
	print(f"\tLibrosa <-> Scipy: {int16_match1}")
	print(f"\tLibrosa <-> Tensorflow: {int16_match2}")
	print(f"\tScipy <-> Tensorflow: {int16_match3}")
	print("-" * 72)

	# -----------------------------------------------------------------
	# Test the different wasy of taking a wav file and converting it to
	# mel-spectrograms.
	print("CONVERT AUDIO TO MEL SPECTROGRAM:")



	print("-" * 72)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
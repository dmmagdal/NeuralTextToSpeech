# main.py
# Run through the different modules and ways to read (wav) audio files
# and convert the data to mel-spectrograms the same way done in librosa
# or the PyTorch Tacotron STFT module.


import numpy as np
import tensorflow as tf
import torch
from layers import TacotronSTFT
from load_audio import load_wav_scipy, load_wav_librosa, load_wav_tf
from process_audio import get_mel_librosa, get_mel_spec_tf
from audio_processing_tf import STFT


def main():
	# Hyperparameters & variables.
	path = "./LJ001-0001.wav"
	max_wav_value = 32768.0
	sampling_rate = 22050
	filter_length = 1024
	hop_length = 256
	win_length = 1024
	n_mel_channels = 80
	mel_fmin = 0.0
	mel_fmax = 8000.0

	# -----------------------------------------------------------------
	# Test the different ways of loading a wav file.
	print("READ AUDIO FILE:")
	print("-" * 32)

	# Scipy.
	scipy_audio, scipy_sr = load_wav_scipy(path)
	print(f"Scipy read shape: {scipy_audio.shape}")

	# Librosa.
	librosa_audio, librosa_sr = load_wav_librosa(path)
	print(f"Librosa read shape: {librosa_audio.shape}")

	# Tensorflow.
	tf_audio, tf_sr = load_wav_tf(path)
	print(f"Tensorflow read shape: {tf_audio.shape}")
	print("-" * 32)

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
	print("-" * 32)

	# PyTorch Tacotron 2 STFT (baseline).
	# Convert scipy audio data to a float32 pytorch tensor (as seen in
	# utils.py from the Nvidia Tacotron2 repo).
	stft = TacotronSTFT(
		filter_length, hop_length, win_length, n_mel_channels, 
		sampling_rate, mel_fmin, mel_fmax
	)
	pytorch_audio = torch.FloatTensor(scipy_audio.astype(np.float32))
	audio_norm = pytorch_audio / max_wav_value
	audio_norm = audio_norm.unsqueeze(0)
	audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
	melspec = stft.mel_spectrogram(audio_norm)
	pytorch_melspec = torch.squeeze(melspec, 0)
	print(f"PyTorch (Tacotron2 STFT) mel-spec shape: {pytorch_melspec.shape}")

	# Tensorflow.
	tf_melspec = get_mel_spec_tf(
		path, filter_length, hop_length, win_length, n_mel_channels, 
		sampling_rate, mel_fmin, mel_fmax
	)
	print(f"Tensorflow mel-spec shape: {tf_melspec.shape}")

	# Librosa.
	librosa_melspec = get_mel_librosa(
		path, filter_length, hop_length, win_length, n_mel_channels
	)
	print(f"Librosa mel-spec shape: {librosa_melspec.shape}")
	print("-" * 32)

	# All data arrays (converted to numpy) are the same.
	tf_melspec = tf.transpose(tf_melspec, [1, 0])
	melspec_match1 = np.array_equal(pytorch_melspec.numpy(), librosa_melspec)
	melspec_match2 = np.array_equal(librosa_melspec, tf_melspec.numpy())
	melspec_match3 = np.array_equal(pytorch_melspec.numpy(), tf_melspec.numpy())

	# Output comparisons.
	print(f"Data matching (raw):")
	print(f"\tLibrosa ({librosa_melspec.dtype}) <-> PyTorch ({pytorch_melspec.numpy().dtype}): {melspec_match1}")
	print(f"\tLibrosa ({librosa_melspec.dtype}) <-> Tensorflow ({tf_melspec.numpy().dtype}): {melspec_match2}")
	print(f"\tPytorch ({pytorch_melspec.numpy().dtype}) <-> Tensorflow ({tf_melspec.numpy().dtype}): {melspec_match3}")

	# Pad Tensorflow tensor since it is shorter than the others.
	n_pad = pytorch_melspec.shape[-1] - tf_melspec.shape[-1]
	tf_melspec = tf.pad(tf_melspec, [[0, 0], [0, n_pad]])

	# Compute the L1 difference between all data arrays.
	l1_loss = torch.nn.L1Loss()
	melspec_l1_1 = l1_loss(torch.tensor(librosa_melspec), pytorch_melspec)
	melspec_l1_2 = l1_loss(torch.tensor(librosa_melspec), torch.tensor(tf_melspec.numpy()))
	melspec_l1_3 = l1_loss(pytorch_melspec, torch.tensor(tf_melspec.numpy()))

	# Output L1 comparisons.
	print(f"L1 (MAE) difference:")
	print(f"\tLibrosa <-> PyTorch: {melspec_l1_1}")
	print(f"\tLibrosa <-> Tensorflow: {melspec_l1_2}")
	print(f"\tPytorch <-> Tensorflow: {melspec_l1_3}")

	# Compute the L2 difference between all data arrays.
	l2_loss = torch.nn.MSELoss()
	melspec_l2_1 = l2_loss(torch.tensor(librosa_melspec), pytorch_melspec)
	melspec_l2_2 = l2_loss(torch.tensor(librosa_melspec), torch.tensor(tf_melspec.numpy()))
	melspec_l2_3 = l2_loss(pytorch_melspec, torch.tensor(tf_melspec.numpy()))

	# Output comparisons.
	print(f"L2 (MSE) difference:")
	print(f"\tLibrosa <-> PyTorch: {melspec_l2_1}")
	print(f"\tLibrosa <-> Tensorflow: {melspec_l2_2}")
	print(f"\tPytorch <-> Tensorflow: {melspec_l2_3}")

	# Fully process the data in Tensorflow just how it is done in
	# data.py (the data loading module in all active model folders).
	# This is meant to also more closely follow the data loading done
	# in Tacotron 2.
	tf_STFT = STFT(
		filter_length, win_length, hop_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax
	)
	tf_audio_norm = tf_audio / max_wav_value
	# tf_audio_norm = tf.expand_dims(tf_audio_norm, 0)
	tf_stft_melspec = tf_STFT.mel_spectrogram(tf_audio_norm)

	tf_stft_melspec = tf.transpose(tf_stft_melspec, [1, 0])

	# Pad Tensorflow tensor since it is shorter than the others.
	n_pad = pytorch_melspec.shape[-1] - tf_stft_melspec.shape[-1]
	tf_stft_melspec = tf.pad(tf_stft_melspec, [[0, 0], [0, n_pad]])

	# Recalculate L1 difference between the tensorflow data array and 
	# the rest.
	melspec_l1_tf_2 = l1_loss(torch.tensor(librosa_melspec), torch.tensor(tf_stft_melspec.numpy()))
	melspec_l1_tf_3 = l1_loss(pytorch_melspec, torch.tensor(tf_stft_melspec.numpy()))

	# Output comparisons.
	print(f"L1 (MAE) difference (Tensorflow STFT):")
	print(f"\tLibrosa <-> Tensorflow: {melspec_l1_tf_2}")
	print(f"\tPytorch <-> Tensorflow: {melspec_l1_tf_3}")

	# Recalculate L2 difference between the tensorflow data array and 
	# the rest.
	melspec_l2_tf_2 = l2_loss(torch.tensor(librosa_melspec), torch.tensor(tf_stft_melspec.numpy()))
	melspec_l2_tf_3 = l2_loss(pytorch_melspec, torch.tensor(tf_stft_melspec.numpy()))

	# Output comparisons.
	print(f"L2 (MSE) difference (Tensorflow STFT):")
	print(f"\tLibrosa <-> Tensorflow: {melspec_l2_tf_2}")
	print(f"\tPytorch <-> Tensorflow: {melspec_l2_tf_3}")
	print("-" * 72)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
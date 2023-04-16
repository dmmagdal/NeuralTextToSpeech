# utils.py

import glob
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectrogram(spectrogram):
	fig, ax = plt.subplots(figsize=(10, 2))
	im = ax.imshow(
		spectrogram, aspect="auto", origin="lower",
		interpolation='none'
	)
	plt.colorbar(im, ax=ax)

	fig.canvas.draw()
	plt.close()

	return fig


def init_weights(mean=0.0, std=0.1):
	return tf.keras.initializers.RandomNormal(mean, std)


def get_padding(kernel_size, dilation=1):
	padding = int((kernel_size * dilation - dilation) / 2)
	return padding
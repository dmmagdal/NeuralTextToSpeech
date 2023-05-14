# utils.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def intersperse(lst, item):
	# Adds blank symbol.
	result = [item] * (len(lst) * 2 + 1)
	result[1::2] = lst
	return result


def save_figure_to_numpy(fig):
	data = np.fromstring(
		fig.canvas.tostring_rgb(), dtype=np.uint8, sep=''
	)
	data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	return data


def plot_tensor(tensor):
	plt.style.use('default')
	fig, ax = plt.subplots(figsize=(12, 3))
	im = ax.imshow(
		tensor, aspect='auto', origin='lower', interpolation='none'
	)
	plt.colorbar(im, ax=ax)
	plt.tight_layout()
	fig.canvas.draw()
	data = save_figure_to_numpy(fig)
	plt.close()
	return data


def save_plot(tensor, savepath):
	plt.style.use('default')
	fig, ax = plt.subplots(figsize=(12, 3))
	im = ax.imshow(
		tensor, aspect='auto', origin='lower', interpolation='none'
	)
	plt.colorbar(im, ax=ax)
	plt.tight_layout()
	fig.canvas.draw()
	plt.savefig(savepath)
	plt.close()
	return

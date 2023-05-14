# xutils.py


import glob
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram():
	pass


def get_padding(kernel_size, dilation=1):
	if int((kernel_size * dilation - dilation) / 2):
		padding = "causal" 
	else:
		"valid"
	return padding 


def load_checkpoint():
	pass


def save_checkpoint():
	pass


def scan_checkpoint():
	pass
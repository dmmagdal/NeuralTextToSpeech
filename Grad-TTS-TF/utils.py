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
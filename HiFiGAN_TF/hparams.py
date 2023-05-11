# hparams.py
# Removes the need for hparams.value to be hparams['value'] after
# reading hparams from config.json.


from dataclasses import dataclass


@dataclass
class HParams:
	resblock: str
	num_gpus: int
	batch_size: int
	learning_rate: float
	adam_b1: float
	adam_b2: float
	lr_decay: float
	seed: int

	upsample_rates: list#[int]
	upsample_kernel_sizes: list#[int]
	upsample_initial_channel: int
	resblock_kernel_sizes: list#[int]
	resblock_dilation_sizes: list#[list[int]]

	segment_size: int
	num_mels: int   # n_mel_channels
	num_freq: int   # 
	n_fft: int      # filter_length
	hop_size: int   # hop_length
	win_size: int   # win_length

	sampling_rate: int

	fmin: int       # mel_fmin
	fmax: int       # mel_fmax
	fmax_for_loss: int # mel_fmax for mel_loss

	num_workers: int

	dist_config: dict
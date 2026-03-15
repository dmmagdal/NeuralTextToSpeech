# torch_wav2mel.py
# Source: https://github.com/jaywalnut310/vits/blob/main/mel_processing.py
# Source: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/hifi-gan/meldataset.py
# Reference: https://pytorch.org/docs/stable/generated/torch.hann_window.html
# Reference: https://pytorch.org/docs/stable/generated/torch.stft.html
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html


import torch
from librosa.filters import mel as librosa_mel_fn


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, 
		win_size, fmin, fmax, center=False):
	if torch.min(y) < -1.0:
		print("min value is ", torch.min(y))
	if torch.max(y) > 1.0:
		print("max value is ", torch.max(y))

	mel_basis, hann_window = {}, {}
	if fmax not in mel_basis:
		mel = librosa_mel_fn(
			sampling_rate, n_fft, num_mels, fmin, fmax
		)
		mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float()#.to(device)
		hann_window[str(y.device)] = torch.hann_window(win_size)#.to(device)

	y = torch.nn.functional.pad(
		y.unsqueeze(1), 
		(int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
		mode='reflect'
	)
	y = y.squeeze(1)

	spec = torch.stft(
		y, n_fft, hop_length=hop_size, win_length=win_size,
		window=hann_window[str(y.device)], center=center,
		pad_mode="reflect", normalized=False, onesided=True
	)
	spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

	spec = torch.matmul(
		mel_basis[str(fmax) + "_" + str(y.device)], spec
	)
	spec = spectral_normalize_torch(spec)

	return spec


def spectral_normalize_torch(magnitudes):
	output = dynamic_range_compression_torch(magnitudes)
	return output


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
	return torch.log(torch.clamp(x, min=clip_val) * C)
import tensorflow as tf
import tensorflow_io as tfio
from Tacotron2_TF.process_audio import *
from Flowtron_TF.audio_processing_tf import STFT
from scipy.io.wavfile import read


def dynamic_range_compression(x, C=1, clip_val=1e-5):
	"""
	PARAMS
	------
	C: compression factor
	"""
	return tf.math.log(tf.clip_by_value(
		x, clip_value_min=clip_val, clip_value_max=tf.float32.max) * C
   )


# Path to target wav.
path = "./Flowtron_TF/LJSpeech-1.1/wavs/LJ023-0120.wav"

# Mel spectrogram from librosa.
mel_spec_librosa = get_mel_librosa(
	path, n_fft=1024, hop_length=256, win_length=1024, n_mels=80,
)
print("melspec librosa shape: {}".format(mel_spec_librosa.shape))
print(mel_spec_librosa) # Expected shape (80, 783)
mel_spec_librosa = dynamic_range_compression(mel_spec_librosa)
print(mel_spec_librosa)
print("-" * 72)

# Mel spectrogram from tensorflow
stft = STFT(
	filter_length=1024, frame_step=256, frame_length=1024, 
	sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0
)
sampling_rate, data = read(path)
data = tf.convert_to_tensor(data, dtype=tf.float32)
mel_spec_tf = stft.mel_spectrogram(data)
print("melspec tensorflow shape: {}".format(mel_spec_tf.shape))
print(mel_spec_tf) # Expected shape (779, 80)
mel_spec_tf = dynamic_range_compression(mel_spec_tf)
print(mel_spec_tf)
print("-" * 72)

# Note that in TacotronSTFT in Nvidia's Tacotron 2 repo (which is also
# present in other repositories such as Nvidia's Flowtron
# implementation), the output shape from the exact same source file is
# (80, 783). That is the exact same shape as the librosa mel
# spectrogram but is not even close to having the same values. To get
# somewhat close to the values is by passing the mel spectrograms
# through the dynamic compression function.

# Mel spectrogram from tensorflow (alternative)
mel_spec_tf = get_mel_spec_tf(
	path, frame_length=1024, frame_step=256, fft_length=1024, 
	n_mel_channels=80, sample_rate=22050, mel_fmin=0.0, mel_fmax=8000.0
)
print("melspec tensorflow shape: {}".format(mel_spec_tf.shape))
print(mel_spec_tf) # Expected shape (779, 80)
mel_spec_tf = dynamic_range_compression(mel_spec_tf)
print(mel_spec_tf)
print("-" * 72)

# Another note is that by loading the file in via the get_mel_spec_tf()
# function which loads the audio using tensorflow instead of scipy and
# handles the data with the same functions as what is implemented in
# the Tensorflow STFT function. While the shape is still not the same
# as what is produced by the Pytorch Tacotron2 STFT module, the values
# are still MUCH closer than the nearest values provided by the
# get_mel_librosa() function (which uses librosa to load the audio and
# perform the STFT conversion to mel spectrogram).

# FastSpeech2

Description: A Tensorflow implementation of the FastSpeech2 text to speech model in Tensorflow 2.


### Notes:
 * Using alternative way to process audio from wav to mel spectrograms in Tensorflow as seen in Tacotron2_TF and Flowtron_TF folders. Refer to there for more information as well as any discrepencies. 
 * The following variables for audio processing are used interchangeably (along with the defaults specified by the hparams/config file in the Nvidia Tacotron2/Flowtron repository):
 	- sampling_rate (22050)
 	- filter_length = nfft_length (1024)
 	- hop_length = frame_step (256)
 	- win_length = frame_length (1024)
 	- mel_fmin (0.0)
 	- mel_fmax (8000.0)
 	- n_mel_channels (80)
 * While I am keeping the preprocessed_data, I will only be implementing the general preprocessor and the LJSpeech preprocessor as I only wish to train the base model with that dataset and allow for fine-tuning on smaller, generic/less known datasets.
 * There is A LOT of preprocessing to do from the FastSpeech2 repo. For the three datasets that are referenced, the repo uses montreal forced alignment information which is not available for new datasets. The amount of preprocessing work is putting this on the backburning.
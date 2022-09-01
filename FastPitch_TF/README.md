# README

Description: Guide to FastPitch folder in NVIDIA DeepLearningExamples repo (/PyTorch/SpeechSynthesis/FastPitch).


### FastPitch root folder Structure

Ignore the following files/folders from the FastPitch directiory:
 * audio: contains output audio samples (wav) from FastPitch
 * fileslist: contains the fileslists for LJSpeech dataset and the mappings between audio, text, and pitch.
 * hifigan: For loading a HiFi-GAN model. Use the HiFi-GAN training and code in the repository (/PyTorch/SpeechSynthesis/HiFi-GAN).
 * img: provides reference images for the README in the FastPitch folder.
 * notebooks: notebooks for FastPitch inference.
 * platform: scripts for specialized hardware such as the DGX1 or DGXA100.
 * scripts: scripts for executing code to run training/inference on docker.
 * triton: deploying FastPitch model on Triton inference server.
 * waveglow: For loading a WaveGLOW model. Use the WaveGLOW training and code in the following repository (also from NVIDIA): https://github.com/NVIDIA/waveglow

The following files/folders remaining are necessary for the workflow of FastPitch:
 * cmudict: text processing for audio.
 * common:
 * fastpitch: code for the FastPitch model. Also includes dataset initialization.
 * phrases: ???
 * inference.py: multiple processing layers and utility functions.
 * models.py: load models (FastPitch, WaveGLOW, or HiFi-GAN).
 * prepare_dataset.py: preprocess the dataset.
 * train.py: train FastPitch model.

Notes:
 * A single iteration through the training dataset (ljs_audio_text_train_v3.txt) takes 16 hours 45 minutes to complete (this includes all the computation that goes into each feature). Once the data has already been preprocessed (and saved to disk), an iteration now takes up to 45 minutes.
 * When extracting pitch, pitch length is 4 timesteps longer than the mel-spectrogram length (original implementation had a threshold of 1 timestep). This is consistent across all samples.
 * Loading audio from librosa is about the same as loading with tensorflow. Tensorflow is used for this project to keep things streamlined to one library. See [librosa documentation](https://librosa.org/doc/main/generated/librosa.load.html).
 * The [librosa pyin function](https://librosa.org/doc/main/generated/librosa.pyin.html) returns the fundamental frequency (f0) estimation using probabilistic YIN (pYIN). This is then used to try to estimate the pitch.
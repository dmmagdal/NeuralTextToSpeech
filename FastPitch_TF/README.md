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
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
 * When attempting to preprocess the LJSpeech dataset in the original implementation ([repo here](https://github.com/NVIDIA/DeepLearningExamples/tree/3b05bf180881225f24d1a2db1986014d35f1633d/PyTorch/SpeechSynthesis/FastPitch)), the following error would be issued:
 ```
DLL 2022-09-24 12:49:10.200550 - PARAMETER dataset_path : ./
DLL 2022-09-24 12:49:10.200550 - PARAMETER wav_text_filelists : ['.\\filelists\\ljs_audio_text.txt']
DLL 2022-09-24 12:49:10.200550 - PARAMETER extract_mels : False
DLL 2022-09-24 12:49:10.200550 - PARAMETER extract_pitch : False
DLL 2022-09-24 12:49:10.200550 - PARAMETER save_alignment_priors : False
DLL 2022-09-24 12:49:10.200550 - PARAMETER log_file : preproc_log.json
DLL 2022-09-24 12:49:10.200550 - PARAMETER n_speakers : 1
DLL 2022-09-24 12:49:10.200550 - PARAMETER max_wav_value : 32768.0
DLL 2022-09-24 12:49:10.200550 - PARAMETER sampling_rate : 22050
DLL 2022-09-24 12:49:10.201536 - PARAMETER filter_length : 1024
DLL 2022-09-24 12:49:10.201536 - PARAMETER hop_length : 256
DLL 2022-09-24 12:49:10.201536 - PARAMETER win_length : 1024
DLL 2022-09-24 12:49:10.201536 - PARAMETER mel_fmin : 0.0
DLL 2022-09-24 12:49:10.201536 - PARAMETER mel_fmax : 8000.0
DLL 2022-09-24 12:49:10.201536 - PARAMETER n_mel_channels : 80
DLL 2022-09-24 12:49:10.201536 - PARAMETER f0_method : pyin
DLL 2022-09-24 12:49:10.201536 - PARAMETER batch_size : 1
DLL 2022-09-24 12:49:10.201536 - PARAMETER n_workers : 16
Processing .\filelists\ljs_audio_text.txt...
Traceback (most recent call last):
  File ".\prepare_dataset.py", line 174, in <module>
    main()
  File ".\prepare_dataset.py", line 129, in main
    pitch_online_method=args.f0_method)
  File "D:\FastPitch\fastpitch\data_function.py", line 190, in __init__
    self.betabinomial_interpolator = BetaBinomialInterpolator()
  File "D:\FastPitch\fastpitch\data_function.py", line 54, in __init__
    self.bank = functools.lru_cache(beta_binomial_prior_distribution)
  File "C:\Users\magda\AppData\Local\Programs\Python\Python37\lib\functools.py", line 477, in lru_cache
    raise TypeError('Expected maxsize to be an integer or None')
TypeError: Expected maxsize to be an integer or None
 ```
 According to following [GitHub Issue](https://github.com/NVIDIA/DeepLearningExamples/issues/1016) on the repo, this was due to the `functools.lru_cache()` function working differently from Python 3.8+. However, there is a workaround with Python <=3.7 with the following code:
 ```
 f = functools.lru_cache(maxsize=128)
 self.bank = f(beta_binomial_prior_distribution)
 ```
 * The (LJSpeech) dataset is initialized in the `data_function.py` under the `Data` class. The dataset is initialized as a tf.data.dataset by using the `from_generator()` function ([documentation](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator)). There is a way to initialize the dataset with the `from_tensor_slices()` function ([documentation](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices)) but that will cause OOM on my 8GB GPU (probably from all the extra data from calculating/loading the pitch, energy, & attention prior matrices). Using `from_generator()` allows the dataset to be loaded dynamically while `from_tensor_slices()` loads everything to memory. See the linked documentation for any caveats ([Building Tensorflow input pipelines with tf.data](https://www.tensorflow.org/guide/data)).
 * Compiling the model in eager execution mode (set `run_eagerly=True` in the model's [`compile()`](https://www.tensorflow.org/versions/r2.7/api_docs/python/tf/keras/Model#compile) function) will eagerly take the data from the tf.data.dataset (with the specified batch size, especially when the dataset was generated with the `from_generator()` function). This is useful for debugging & tracing the shape and data of tensors throughout the model.
 * Compiling the model in graph execution model (do *not* set the `run_eagerly` parameter in the model's [`compile()`](https://www.tensorflow.org/versions/r2.7/api_docs/python/tf/keras/Model#compile) function) can provide optimizations to make the model run faster with better memory efficiency while eager execution simplifies the model building experience (see this [Tensorflow blog post](https://blog.tensorflow.org/2018/08/code-with-eager-execution-run-with-graphs.html)). We'd like to train the model in graph execution mode for these benefits. One thing to note is that when passing the dataset after initializing it with the `from_generator()` method results in the batch size dimension being None (even when batched & prefetched). Another thing to note about compiling in graph execution, particularly when the dataset is initialized with `from_generator()`, is that the output shapes should be explicitly outlined in the `output_signature` argument of the `from_generator()` function. If the shapes include None (to account for inputs of dynamic length/shape), this may cause errors when the data is passed through the model (the tensor will have None as part of the shape). To avoid this, compute the maximum length for the respective elements of the dataset and save that as a class file to be accessed later.
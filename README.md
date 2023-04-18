# Neural Text-to-Speech

Description: This repository attempts to replicate populate text-to-speech models such as Tacotron 2 in Tensorflow2 for the purposes of simplifying and explaining each model.


### References

#### Tacotron 2
 - [paper](https://arxiv.org/pdf/1712.05884.pdf)
 - [code](https://github.com/NVIDIA/tacotron2)

#### Flowtron
 - [paper](https://arxiv.org/pdf/2005.05957.pdf)
 - [code](https://github.com/NVIDIA/flowtron)

#### FastSpeech 2
 - [paper](https://arxiv.org/pdf/2006.04558.pdf)
 - [code](https://github.com/ming024/FastSpeech2)

#### FastPitch
 - [paper](https://arxiv.org/pdf/2006.06873.pdf)
 - [code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)

#### TalkNet 2
 - [paper](https://arxiv.org/pdf/2104.08189.pdf)
 - [code](https://github.com/NVIDIA/NeMo) (Note: No longer available on the Nvidia Nemo repo. Going through the commit history can find you the last revision of the code before it was removed)

#### Misc
 - Add [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) to the repo
 - Audio Processing:
    - Keras MelGAN [example](https://keras.io/examples/audio/melgan_spectrogram_inversion/#loading-the-dataset)
    - Keras ASR with CTC [example](https://keras.io/examples/audio/ctc_asr/#load-the-ljspeech-dataset)
    - Keras ASR with Transformer [example](https://keras.io/examples/audio/transformer_asr/#preprocess-the-dataset)
    - Process Audio with Tensorflow [medium article](https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06)
 - Create Tensorflow Records [medium article](https://medium.com/nerd-for-tech/how-to-create-tensorflow-tfrecords-out-of-any-dataset-c64c3f98f4f8)
 - Create Tensorflow Records Keras [example](https://keras.io/examples/keras_recipes/creating_tfrecords/)
# DiffWave

Description: DiffWave is a Diffusion vocoder used with neural text to speech models.


### Notes:

 * Tensors in Pytorch would be shaped (batch_size, channels, height, width) while tensors in Tensorflow are shaped (batch_size, height, width, channels). The channels dim/axis is what is operated on by each framework's respective layers so there is a need to transpose the tensors in this tensorflow implementation (I cannot vouch for its correctness but I can say that the model is able to build & run).
 * Added explicit dtype conversions to the `steps` and `dims` tensor rather than let implicit conversion. The implicit conversion results in errors when trying to multiply the `dims` tensor (eager tensor with implicit dtype `tf.int32`) with `4.0` (dtype `tf.float32`). 
 * The `kernel_size` and `strides` are flipped in the `Conv2DTranspose` layers (from the `SpectrogramUpsampler` layer). This is because the dims are flipped on the input mel spectrogram tensor in Tensorflow compared to the implementation in PyTorch.
 * Original repo can be found [here](https://github.com/lmnt-com/diffwave). The original paper for DiffWave is [here](https://arxiv.org/pdf/2009.09761.pdf).
 * The batch size was brought down from 16 to 4 for the sake of being able to train. Also note that modifying the batch size requires modification to the number of steps for training (ie 120,000 steps at batch size 4 != 120,000 steps at batch size 16; 120,000 steps at batch size 4 = 30,000 steps at batch size 16).
 * The large loading bar that comes up when training the model is the data loader bar. The data loader resets at the end of every epoch (probably because the dataset was loaded from generator? Could be dynamically loaded because of it?). The training tracker loading bar is only properly visible at the end of an epoch (and subsequently the end of the data loader bar).
 * There are three sorts of data and two sorts of models available for DiffWave. Unconditional model (and data), conditional model (and data), and gtzan data (use unconditional model). I don't know the difference between all of them but the data appears to have an impact on the shape (I'd have to verify). 
 	 * I have only verified training the conditional model with conditonal data on LJSpeech dataset.
  * Called `Conv1D` layer `DilatedConv1D`. Not sure if that is actually an accurate representation of the layer but seems close enough. This (diffwave repo](https://github.com/revsic/tf-diffwave) implemented in tensorflow refers to it as `DilatedConv1d` (and it also refers to the overall DiffWave model/neural network as Wavenet).


### TODO List (for V1 release)

 [x] Verify model architecture
 [x] Finish and verify training loop
 	 [x] Train step
 	 [x] Validation step
 [x] Model checkpointing, saving, & loading
 [ ] Model resume training
 [ ] Model inference
UPDATE:
 * Eager execution of the model for training OOMs on GPU. On CPU, the model is able to train, but it is not at all viable for actual training.
 * Graph execution of the model for training actually works.


### References

 * DiffWave Repos
	* [Diffwave](https://github.com/lmnt-com/diffwave)
	* [TF-Diffwave](https://github.com/revsic/tf-diffwave)
 * Tensorflow
	 * Audio ([tf.audio](https://www.tensorflow.org/api_docs/python/tf/audio))
		 * [encode_wav](https://www.tensorflow.org/api_docs/python/tf/audio/encode_wav)
		 * [decode_wav](https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav)
	 * File IO ([tf.io](https://www.tensorflow.org/api_docs/python/tf/io))
		 * [read_file](https://www.tensorflow.org/api_docs/python/tf/io/read_file)
		 * [write_file](https://www.tensorflow.org/api_docs/python/tf/io/write_file)
	 * Customize Training
		 * [customize what happens in model.fit()](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit)
		 * [writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
	 * Model Checkpoints, Saving, & Loading
		 * [checkpoint callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
		 * [training checkpoints](https://www.tensorflow.org/guide/checkpoint)
		 * [train checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)
		 * [train latest_checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint)
		 * [save and load models](https://www.tensorflow.org/tutorials/keras/save_and_load)
	 * Tensorboard
		 * [tensorboard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)
		 * [getting started](https://www.tensorflow.org/tensorboard/get_started) with tensorboard
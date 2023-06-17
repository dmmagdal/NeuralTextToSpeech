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
 * Resuming training model from checkpoint:
	 * There are two ways to save the DiffWave model, one is to save the weights (in a H5 file) or save the whole model in a SavedModel or HDF5 format).
	 * When you save the weights of a model, you have to re initialize the model (with the same hyperparameters) before loading the model weights ([documentation](https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights)). Because of this, you get full access to the model functions (ie train_step(), fit(), compute_loss()). You would also lose optimizer state if you wished to continue training/finetuning the model.
	 * When saving a model with tf.keras.Model.save(), there is the option to save the model either in HDF5 (H5) format or SavedModel format (the default).
		 * The SavedModel format is a directory containing a protobuf binary and a TensorFlow checkpoint. When loading a model from SavedModel, the restored model is compiled with the same arguments as the original model ([documentation](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)).
		 * The HDF5 format works similarly to SavedModel. The HDF5 format contains everything in an H5 file ([documentation](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format)).
	 * Keras saves models by inspecting their architectures. This technique saves everything, including the weight values, the model's architecture, the model's training configuration (what you pass to the .compile() method), and the optimizer and its state, if any (this enables you to restart training where you left off).
	 * The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and custom layers without requiring the original code ([documentation](https://www.tensorflow.org/tutorials/keras/save_and_load#saving_custom_objects)).
		 * To save custom objects to HDF5, you must do the following:
			 1. Define a get_config method in your object, and optionally a from_config classmethod.
				 * get_config(self) returns a JSON-serializable dictionary of parameters needed to recreate the object.
				 * from_config(cls, config) uses the returned config from get_config to create a new object. By default, this function will use the config as initialization kwargs (return cls(\*\*config)).
			 2. Pass the object to the custom_objects argument when loading the model. The argument must be a dictionary mapping the string class name to the Python class. E.g. tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
	 * When you load a SavedModel in Python, all tf.Variable attributes, tf.function-decorated methods, and tf.Modules are restored in the same object structure as the original saved tf.Module ([documentation](https://www.tensorflow.org/guide/saved_model#loading_and_using_a_custom_model), [serialization documentation](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md#serialization-details)).
		 * The SavedModel format is not the same as tf.keras.Model. This means that some functions like call(), predict(), and summary() still work just fine, but other functions like train_step(), compute_loss(), and fit() are not available to use once loaded. This means that in order to retrain or finetune a model loaded from SavedModel, a custom training loop must be used instead of model.fit() + model.train_step()/model.valid_step(). In addition, model attriutes (ie self.params or self.is_conditional) are not accessible from the SavedModel format since they were saved in the sub-classed tf.keras.Model class (supposedly, you can access variables that were declared tf.Variable() inside the SavedModel).
		 * Another note about SavedModels, because they require a custom training loop outside of Model.train_step(), there is a crucial caveat regarding resources on the host system. Model.fit() has optimizations built in that allow it to be more memory efficient. Custom training loops, while still being written in tensorflow, don't have those optimizations. Hence, there is a chance of OOM'ing on a machine with smaller resources
			 * Currently, training will not OOM with Model.fit() but will OOM on a custom training loop on my 2060 SUPER (8GB VRAM). The only way to not OOM on the custom training loop that I've seen is by wrapping the code/function with `@tf.function` to run that code in graph execution mode.
 * About eager vs graph execution:
	 * Tensorflow runs in eager execution by default.
	 * To run code in graph execution, use the `@tf.function` decorator to the top of the function.
	 * In `tf.keras.Model.compile()`, the `run_eagerly` argument details whether the model is wrapped in `tf.function`. By default, that value is False (`tf.function` wraps around the model code, and the model is run in graph execution). When the value is set to True, `tf.function` is NOT wrapped around the model and the model code is run in eager execution.
 * Functional API vs Sub-classed tf.keras.Model
	 * The base DiffWave model (which can be considered a WaveNet model) takes in multiple inputs:
		 1. the raw audio signal (shape (None,)).
		 2. the diffusion timestep (shape ()).
		 3. the mel spectrogram signal (shape (None, n_mels)). This input is optional if using the unconditonal (or gtzan) model.
	 * A sub-classed tf.keras.Model does not like to pass in multiple inputs to Model.call() (e.g. Model.call(input_1, input_2, input_3, training=None)). To pass in multiple inputs, they would have to be packaged in a tuple (Model.call((input_1, input_2, input_3), training=None)). Optional inputs could be passed in as part of the \*\*kwargs after the training=None value (Model.call((input_1, input_2), training=None, input_3=None)). There is a problem with this though, it makes the model very hard to build (call Model.build()) and resume training or run inference on a saved model (use Model.call() with training=False or True). This created numerous errors, regardless of the save formats used (SavedModel, SavedModel with h5, and save weights with h5). This also applied to attempts at retraining or finetuning the model (both with Model.fit() and a custom training loop).
	 * Using the Functional API from tensorflow provides a lot more flexibility in terms of being able to work with multiple inputs to a model. Multiple inputs can be specified with tf.keras.layers.Layer.Input() layers. For the optional inputs, some conditional logic could be applied but note that will affect the input of the model (especially when you consider the different layers and inputs that are included in the conditional Diffwave model vs the unconditional Diffwave model). Working with the functional API has allowed the model to be built (Model.build()/Model.summary()), resume training (load from from SavedModel checkpoint and use Model.call(input, training=True) in custom training loop), and inference (Model.predict(), Model.predict_on_batch(), Model(); Model.call(input, training=False) does seem to have an issue which I'll debug later). The Functional API also requires the use of a custom training loop instead of defining Model.train_step() if required.
 * Training Notes:
	 * Original authors trained their model for 1 million steps at batch size 16 for 13,000 to 16,000 samples across 8x Nvidia 2070Ti cards (8GB VRAM each). 
		 * To match that training, I trained this model for 4 million steps at batch size of 8 (equivalent to 1 million steps at batch size 16) due to the original batch size being too much for my GPU (I would get OOM errors from the graphics card). This was on the LJSpeech dataset (13,000 audio samples).
	 * 1 epoch of training on my machine at the above settings takes anywhere between 18 to 20 minutes. To train the full model (without interruptions) would take over 2 weeks (14+ days).
		 * Because an epoch would take so long to train, I have my checkpoint callback save at every epoch. I would have liked to have been saving every 5 or 10 epochs but the cost of making up time between the lost epochs would have lengthened the training time in the event of interruptions (which did happen).
	 * Training on my Desktop (Nvidia 2060 SUPER 8GB) crashes roughly every 100+ epochs due to one reason or another. Root cause isn't clear other than possible memory issues (RAM or VRAM). It always fails when calling the `on_epoch_end()` function on the callback list.
		 * Training started May 31, 2023 at 5:20 PM and ended June 17, 2023 at 11:15 AM.
		 * Training was for 4 million steps at batch size of 4 (1,281 epochs).
		 * Each checkpoint (in SavedModel format) is 36.2MB large. Multiplied by the number of epochs for pretraining on the LJSpeech dataset (1,281 epochs) and the folder containing the checkpoints would be 42.8 GB of data. To reduce the overhead on GitHub (repos have size limits), only every 25th checkpoint is being kept. This brings the count down from 1,281 to 52 checkpoints saved (total storage is now 1.73 GB).
		 * Training had to be resumed 8 times (not including initial start) due to interruptions (roughly every 2 to 3 days).


### TODO List (for V1 release)

 [x] Verify model architecture
 [x] Finish and verify training loop
 	 [x] Train step
 	 [x] Validation step
 [x] Model checkpointing, saving, & loading
 [x] Model resume training
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
		 * [custom training walkthrough](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
	 * Model Checkpoints, Saving, & Loading
		 * [checkpoint callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
		 * [training checkpoints](https://www.tensorflow.org/guide/checkpoint)
		 * [train checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)
		 * [train latest_checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/latest_checkpoint)
		 * [train checkpoint manager](https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager)
		 * [save and load models](https://www.tensorflow.org/tutorials/keras/save_and_load)
		 * [using the savedmodel format](https://www.tensorflow.org/guide/saved_model)
		 * [savedmodel serialization details](https://github.com/tensorflow/community/blob/master/rfcs/20190509-keras-saved-model.md#serialization-details)
	 * Tensorboard
		 * [tensorboard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)
		 * [getting started](https://www.tensorflow.org/tensorboard/get_started) with tensorboard
	 * Callbacks
		 * [callback list](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CallbackList)
		 * [tensorboard](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)
		 * [model checkpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
		 * [early stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
	 * Graph vs Eager Execution
		 * [introduction to graphs and tf.function](https://www.tensorflow.org/guide/intro_to_graphs)
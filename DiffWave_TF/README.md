# DiffWave

Description: Tensorflow implementation of the DiffWave vocoder, a diffusion based vocoder.


### Notes:
 * Added explicit dtype conversions to the `steps` and `dims` tensor rather than let implicit conversion. The implicit conversion results in errors when trying to multiply the `dims` tensor (eager tensor with implicit dtype `tf.int32`) with `4.0` (dtype `tf.float32`). 
 * The `kernel_size` and `strides` are flipped in the `Conv2DTranspose` layers (from the `SpectrogramUpsampler` layer). This is because the dims are flipped on the input mel spectrogram tensor in Tensorflow compared to the implementation in PyTorch.


### References:

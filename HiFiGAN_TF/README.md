# README

Description: HiFi-GAN is a GAN based vocoder used with neural text to speech models.

### Notes:

 * Tensors in Pytorch would be shaped (batch_size, channels, height, width) while tensors in Tensorflow are shaped (batch_size, height, width, channels). The channels dim/axis is what is operated on by each framework's respective layers so there is a need to transpose the tensors in this tensorflow implementation (I cannot vouch for its correctness but I can say that the model is able to build & run).


### Notes

 * Changes from the [Original Repo](https://github.com/jik876/hifi-gan):
     * The training and validation files are specified via the arguments passed into `train.py`. I swapped that out for the setup that I have in GradTTS & FastPitch, where the files are in a folder called 'filelists/` and read from there.
     * `mel_dataset.py` is an attempt to emulate the original data loading function from PyTorch. Given that the original uses `scipy.io.wavfile.read()` to read in the audio data, there will be some slight differences in the data when compared to using tensorflow to read in the audio (as shown in the `AudioProcessing/` folder). Thus, there isn't really a reason to use it over the `data.py` module that's been create and used across the other audio models.


### TODO List (for V1 release)

 1. Finish Model architecture
     * Iron out the padding for the conv layers
     * Go back and add weight initializer for respective layers
     * Implement initialization, training loop, model saving/loading, and compiling in `gan.py`. Alternatively, implement everything in `train.py`
 2. Implement data loading
 3. Implement everything required for training (this can be done either in `gan.py` or `train.py`)
     * Model initialization
     * Model compilation
         * Model loss
         * Model optimizer
     * Training loop
     * Model saving/loading (includes resume saving from epoch)
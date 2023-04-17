# README

Description: HiFi-GAN is a GAN based vocoder used with neural text to speech models.

### Notes:

 * Tensors in Pytorch would be shaped (batch_size, channels, height, width) while tensors in Tensorflow are shaped (batch_size, height, width, channels). The channels dim/axis is what is operated on by each framework's respective layers so there is a need to transpose the tensors in this tensorflow implementation (I cannot vouch for its correctness but I can say that the model is able to build & run).


### Notes


### TODO List (for V1 release)

 1. Finish Model architecture
     * Iron out the padding for the conv layers
     * Go back and add weight initializer for respective layers
     * Implement initialization, training loop, model saving/loading, and compiling in `gan.py`. Alternatively, implement everything in `train.py`
 2. Implement everything required for training (this can be done either in `gan.py` or `train.py`)
     * Model initialization
     * Model compilation
         * Model loss
         * Model optimizer
     * Training loop
     * Model saving/loading (includes resume saving from epoch)
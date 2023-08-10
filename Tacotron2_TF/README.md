# README

Description: Guide to Tacotron2.


### Tacotron root folder Structure

### Notes:

 * A single iteration through the training dataset takes around 20 to 30 minutes to complete (this includes all the computation that goes into each feature). 
 * get_mask_from_lengths() function can be accomplished by tf.sequence_mask(). See documentation [here](https://www.tensorflow.org/api_docs/python/tf/sequence_mask).
 * For the initial embedding layer in Tacotron 2, simply initialize a [tf.keras.layers.Embedding layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) and set the embeddings_initializer argument (default is 'uniform') to a [tf.keras.initializers.RandomUniform(-val, val)](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomUniform) where the minval and maxval arguments are -val and val respectively. This should replicate the embedding layer initialization in the original Tacotron 2 implementation. Some relevant resource links for that include the [pytorch embeddings layer](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) page, [pytorch tensor uniform](https://pytorch.org/docs/stable/generated/torch.Tensor.uniform_.html) page, as well as the following [stack overflow](https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch) post regarding initializing embedding weights.
 * Training calls to layers with LSTMs inside them call [torch.nn.utils.rnn.pad_packed_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html) as well as [torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) which are used for preprocessing data for input to pytorch LSTM layers in training and postprocessing the output. There is no need for such functions in Tensorflow.
     * [pytorch forum on calling flatten_parameters](https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506/2)
     * [pytorch nn.LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
     * stack overflow response on [why sequences are packed in pytorch](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch)
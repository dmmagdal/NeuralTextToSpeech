# save_model_multi_input.py
# Test whether one can save and load a model that uses multiple inputs.


import random
import tensorflow as tf
from tensorflow import keras


# Notes:
# - Attempting to add extra input arguments to the call() function as 
#	follows (def call(self, input1, input2, training=None)), results in
#	errors in tensorflow. It is advised that rather than provide extra
#	inputs that way, they are provided as such 
#	(call(self, inputs, training=None). The inputs are then unpacked
#	within the call() method (inputs1, ..., inputsn = inputs). The
#	model can be built with a call to the build() method, where the
#	input shape is a list of all expected input tensor shapes (defined
#	either via a tuple or TensorShape). For example,
#	(model.build([(None,), (None,)])). Note that for the list of
#	tuples, the inputs are expected to have dtype=tf.float32 by 
#	default.
# - One cannot pass in None to clearly defined input tensors. There is
#	no "default" value for these unless they are set as an optional
#	argument after the training=None argument.


# Used sub-classed model and training defined here: 
# https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e

# Dataset (CIFAR-10)
batch_size = 16
epochs = 5
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# train set / data 
x_train = x_train.astype('float32') / 255

# validation set / data 
x_test = x_test.astype('float32') / 255

# target / class name
class_names = [
	'airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 
	'horse', 'ship', 'truck'
]

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
# validation set / target 
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)


# Define a custom model
class ModelSubClassing(tf.keras.Model):
	def __init__(self, num_classes):
		super(ModelSubClassing, self).__init__()
		# define all layers in init
		# Layer of Block 1
		self.conv1 = tf.keras.layers.Conv2D(
			32, 3, strides=2, activation="relu"
		)
		self.max1  = tf.keras.layers.MaxPooling2D(3)
		self.bn1   = tf.keras.layers.BatchNormalization()

		# Layer of Block 2
		self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu")
		self.bn2   = tf.keras.layers.BatchNormalization()
		self.drop  = tf.keras.layers.Dropout(0.3)

		# GAP, followed by Classifier
		self.gap   = tf.keras.layers.GlobalAveragePooling2D()
		self.dense = tf.keras.layers.Dense(num_classes)


	def call(self, input_tensor, training=False, scaler=1.0):
		# Test to see if list of inputs can be passed.
		input_tensor, x = input_tensor

		# Test to see if training can divert logic.
		if training:
			input_tensor = input_tensor * scaler

		# forward pass: block 1 
		x = self.conv1(input_tensor)
		x = self.max1(x)
		x = self.bn1(x)

		# forward pass: block 2 
		x = self.conv2(x)
		x = self.bn2(x)

		# droput followed by gap and classifier
		x = self.drop(x)
		x = self.gap(x)
		return self.dense(x)


# Instantiate the model
model = ModelSubClassing(num_classes=len(class_names))

# Build the model
# model.build(((None,), (None,))) # Invalid input. Can be list of tensors.
model.build([(None, None, None, 3), (None, None,)]) # Include extra None for batch size I guess
model.summary()

# Train the model (use custom training loop)...
# ...
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(
	optimizer=optimizer, loss=loss_fn, metrics=['accuracy']
)
# model.fit(dataset, epochs=15)
for epoch in range(epochs): # <----- start for loop, step 1
	print(f"Epoch {epoch + 1}/{epochs}")

	# <-------- start for loop, step 2
	# Iterate over the batches of the dataset.
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

		# <-------- start gradient tape scope, step 3
		# Open a GradientTape to record the operations run
		# during the forward pass, which enables auto-differentiation.
		with tf.GradientTape() as tape:

			# Run the forward pass of the layer.
			# The operations that the layer applies
			# to its inputs are going to be recorded
			# on the GradientTape.
			r = tf.random.normal((120,))
			logits = model([x_batch_train, r], training=True) # <- step 4

			# Compute the loss value for this minibatch.
			loss_value = loss_fn(y_batch_train, logits)  # <- step 4 


		# compute the gradient of weights w.r.t. loss  <-------- step 5
		# Use the gradient tape to automatically retrieve
		# the gradients of the trainable variables with respect to the loss.
		grads = tape.gradient(loss_value, model.trainable_weights)

		# update the weight based on gradient  <---------- step 6
		# Run one step of gradient descent by updating
		# the value of the variables to minimize the loss.
		optimizer.apply_gradients(zip(grads, model.trainable_weights))


dummy = tf.random.normal((120,))
input_dummy = list(train_dataset.as_numpy_iterator())[0][0]
model([input_dummy, dummy], scaler=2.0) # Test if extra argument can be passed in.
model.summary()

# Save the model to SavedModel format
model.save('my_custom_model')

# Load the model from SavedModel format
x = tf.keras.models.load_model('my_custom_model')
x.summary()
print(x)
print(dir(x).index('call'))
print(dir(x).index('train_step'))
print(dir(x).index('fit'))

# Try an pass data into loaded model. Set training to true and use 
# additional (optional) values.
x([input_dummy, dummy], scaler=1.0, training=True)
x([input_dummy, dummy]) # Inference mode.

# Try pass in empty value in input tensor list. This will cause an 
# error.
# x([input_dummy, _], scaler=1.0)

# Try pass in None value in input tensor list. This will cause an
# error.
# x([input_dummy, None], scaler=1.0)

print(f"TF version: {tf.__version__}")
print(f"TF executing eagerly {tf.executing_eagerly()}")
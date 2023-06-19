# save_tfa_model.py
# Test whether one can save and load a model that uses layers from 
# tensorflow addons.

import tensorflow as tf
import tensorflow_addons as tfa

# Define a custom model
class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		#self.conv = tfa.layers.GroupNormalization()
		self.layer1 = tf.keras.layers.Dense(32)
		self.layer2 = tfa.layers.GELU()
		self.layer3 = tf.keras.layers.Dense(1, activation="sigmoid")

	def call(self, inputs):
		#return self.conv(inputs)
		x = self.layer1(inputs)
		x = self.layer2(x)
		return self.layer3(x)

# Instantiate the model
model = MyModel()

# Build the model
model.build((32, 32))
model.summary()

# Train the model...
# ...

model(tf.random.normal((32, 32)))

# Save the model to SavedModel format
model.save('my_model')

# Load the model from SavedModel format
x = tf.keras.models.load_model('my_model')
x.summary()

print(f"TF version: {tf.__version__}")
print(f"TF-addons version: {tfa.__version__}")
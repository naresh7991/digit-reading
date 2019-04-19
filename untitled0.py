
#import all important module
import numpy 
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.externals import joblib
from tensorflow.keras.models import model_from_json
import os
#load mnist data
mnist = tf.keras.datasets.mnist
#break data in train test 
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
print(X_train.shape)
#creating the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#model metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#fitting the  model
model.fit(X_train, Y_train, epochs=5)
#evaluating the performance
model.evaluate(X_test, Y_test)

#saving the model 
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# saving  weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


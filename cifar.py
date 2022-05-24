import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

import timeit
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

start = timeit.default_timer()

# setting class names
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train/255.0
x_train.shape

x_test=x_test/255.0
x_test.shape



cifar10_model=tf.keras.models.Sequential()
# First Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same", activation="relu", input_shape=[32,32,3]))
# Second Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same", activation="relu"))
# Max Pooling Layer
cifar10_model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))
# Third Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))
# Fourth Layer
cifar10_model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same", activation="relu"))
# Flattening Layer
cifar10_model.add(tf.keras.layers.Flatten())
# Droput Layer
cifar10_model.add(tf.keras.layers.Dropout(0.5,noise_shape=None,seed=None))
# Adding the first fully connected layer
cifar10_model.add(tf.keras.layers.Dense(units=128,activation='relu'))

# Output Layer
cifar10_model.add(tf.keras.layers.Dense(units=10,activation='softmax'))

cifar10_model.summary()

cifar10_model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])
cifar10_model.fit(x_train,y_train,epochs=15)

test_loss, test_accuracy = cifar10_model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_accuracy))

stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in (seconds): "+str(execution_time))


model.save('my_cifar.h5')
print("Model saved")




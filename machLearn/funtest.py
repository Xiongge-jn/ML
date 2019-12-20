
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from  tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import gzip
train_images = 'MNIST_data/train-images-idx3-ubyte.gz'
# 训练集标签文件
train_labels = 'MNIST_data/train-labels-idx1-ubyte.gz'

# 测试集文件
test_images = 'MNIST_data/t10k-images-idx3-ubyte.gz'
# 测试集标签文件
test_labels = 'MNIST_data/t10k-labels-idx1-ubyte.gz'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding
train_data = extract_data(train_images, 60000)
train_labels = extract_labels(train_labels, 60000)
test_data = extract_data(test_images, 10000)
test_labels = extract_labels(test_labels, 10000)

x_train=train_data.reshape(60000,28,28,1)
y_train=train_labels
x_test=test_data.reshape(10000,28,28,1)
print(x_test.shape)
y_test=test_labels

'''

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(784,)),
    
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
'''
model = tf.keras.models.Sequential([
    Conv2D(32,(5,5),padding='Same', activation='relu',input_shape=(28,28,1)),
    Conv2D(10, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, verbose =2,batch_size=32,epochs=10)

model.evaluate(x_test,  y_test, batch_size=32)
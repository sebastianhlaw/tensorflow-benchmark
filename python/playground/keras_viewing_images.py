# Based on <https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py>

from __future__ import print_function

import math
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

PLOT_DIR = os.path.join(os.path.expanduser("~"), "Development", "tensorflow-benchmark", "data", "plots")


def primes(n):
    """Calculate primes in n.
    Source: <https://rosettacode.org/wiki/Factors_of_an_integer#Python>
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def grid_dimensions(n):
    """Pick as close to square a grid as possible.
    Source: <https://github.com/grishasergei/conviz/blob/master/utils.py>
    """
    factors = primes(n)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]
    i = len(factors) // 2
    return factors[i], factors[i]


def plot_image(images, image=0, save=False):
    """Display an image.
    Args:
        images: Tensor of dimension (?, n, m, channels=1) containing images.
        image: Integer, index of image desired.
        save: Boolean option to save image to disc.
    Returns:
        Nothing. Plots an image, optionally saves to disc.
    """
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[image, :, :, 0], cmap='')
    plt.show()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "images, input-" + str(image).zfill(5) + ".png"),
                    bbox_inches='tight')


def plot_convolution_filters(model, layer, save=False):
    """Plots the convolution filters/kernels for a given layer.
    
    Args:
        model: Keras model object.
        layer: Integer
        save: Boolean option to save image to disc.
    Returns:
        Nothing. Plots an image, optionally saves to disc.
    """
    weights = model.get_weights()
    w = weights[layer]
    w_min = np.min(w)
    w_max = np.max(w)
    n_filters = w.shape[3]
    grid_rows, grid_cols = grid_dimensions(n_filters)
    fig, axes = plt.subplots(min([grid_rows, grid_cols]), max([grid_rows, grid_cols]))
    for f, axis in enumerate(axes.flat):
        im = w[:, :, 0, f]
        axis.imshow(im, vmin=w_min, vmax=w_max,
                    interpolation='nearest', cmap='magma')
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "filters, layer-" + str(layer) + ".png"),
                    bbox_inches='tight')


def plot_convolution_outputs(model, layer, images, image=0, save=False):
    """Plot the images resulting from the application of convolution filters.
    Resources:
        Hints on how to get via Keras <https://www.youtube.com/watch?v=5tW3y7lm7V0>
    Args:
        model: Keras model object.
        layer: Integer, 
        images: Tensor of dimension (?, n, m, channels=1) containing images.
        image: Integer, index of image desired.
        save: Boolean option to save image to disc.
    Returns:
        Nothing. Plots an image, optionally saves to disc.
    """
    in_image = images[image:image + 1, :, :, :]
    out_layer = model.layers[layer].get_output_at(0)
    out_func = K.function([model.layers[0].get_input_at(0)], [out_layer])
    out_image = out_func([in_image])
    n_filters = out_image[0].shape[3]
    grid_rows, grid_cols = grid_dimensions(n_filters)
    fig, axes = plt.subplots(min([grid_rows, grid_cols]), max([grid_rows, grid_cols]))
    for f, axis in enumerate(axes.flat):
        im = out_image[0][0, :, :, f]
        axis.imshow(im, interpolation='nearest', cmap='viridis')
        axis.set_xticks([])
        axis.set_yticks([])
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "images, layer" + str(layer) + ", image-" + str(image).zfill(5) + ".png"),
                    bbox_inches='tight')


# Reproducibility controls
# TODO: Fix reproducibility
# tf.set_random_seed(0)
# np.random.seed(0)
# rn.seed(0)
# <https://github.com/fchollet/keras/issues/2280>
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

# core dimensions
num_classes = 10
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Speed up the process for experimentation by only using a small subset
training_samples = 6000
(x_train, y_train) = (x_train[:training_samples], y_train[:training_samples])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build the model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(7 * 7 * 32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

result = model.fit(x_train, y_train,
                   batch_size=20,
                   epochs=1,
                   verbose=1,
                   validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test,
                       verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

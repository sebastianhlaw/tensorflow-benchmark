# MNIST for ML beginners
# https://www.tensorflow.org/get_started/mnist/pros

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# Function to initialise a tensor of weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Function to initialise a tensor of biases
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set up the model
x = tf.placeholder(tf.float32, [None, 784])  # None indicates dimension can be any length
W = tf.Variable(tf.zeros([784, 10]))  # weights
b = tf.Variable(tf.zeros([10]))  # biases
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])


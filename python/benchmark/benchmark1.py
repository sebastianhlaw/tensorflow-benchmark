"""A benchmark TensorFlow NN MNIST classifier model.

This employs a simple (poor) architecture for classifying MNIST digits.
Architecture:
 Input layer (784 greyscale image -> 28x28)
 Convolution layer 1 (3x3 filter, stride 1, ReLU non-linearity) 
 Pooling layer 1 (2x2 max pool -> 14x14)
 Convolution layer 2 (3x3 filter, stride 1, ReLU non-linearity) 
 Pooling layer 2 (2x2 max pool -> 7x7)
 Fully-connected layer 1
 Output
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Random seed setting for reproducible results
tf.set_random_seed(0)  # This is graph-level seed, needs to be set before graph operations executed.
np.random.seed(0)  # Input data shuffle is contolled by numpy seed.
op_level_seed = None  # No need to set op-level seed if graph-level is set, else weights all initialize identically.

# Dimensions of data
X_SIDE = 28
X_DIM = X_SIDE * X_SIDE
Y_DIM = 10

# Import data
DATA_DIR = "../../data/mnist/"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Input placeholders
x = tf.placeholder(tf.float32, [None, X_DIM])
y_ = tf.placeholder(tf.float32, [None, Y_DIM])
x_square = tf.reshape(x, [-1, X_SIDE, X_SIDE, 1])


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=op_level_seed)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolution_layer(x, shape, activation=tf.nn.relu):
    """Create 2D convolution layer, based on shape input, with 1,1 strides.
    
    Args:
        x: A Tensor
        shape: A shape of [height, width, input channels, output channels]
        activation: An activation/non-linearity function object
    Returns:
        A Tensor
    """
    weights = weight_variable(shape)
    biases = bias_variable([shape[-1]])
    convolution = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    return activation(convolution + biases)


def pooling_layer(x, side, pool=tf.nn.max_pool):
    """Create a pooling layer, with no overlaps
    
    Args:
        x: A Tensor
        side: Side length of pooling square
        pool: A pooling function
        out_dim:
    Returns:
        A Tensor
    """
    return pool(x, ksize=[1, side, side, 1], strides=[1, side, side, 1], padding='SAME')


def fc_layer(x, shape, activation=tf.identity):
    """Create a fully-connected layer, optionally add non-linearity/activation.
    
    Args:
        x: A tensor
        shape: A shape of dimension [input_length, output_length]
        activation: An activation/non-linearity function object, including tf.identity
    Returns:
        A Tensor
    """
    W = weight_variable(shape)
    b = bias_variable([shape[1]])
    return activation(tf.matmul(x, W) + b)


def feed_dict(train, batch_size=10):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(batch_size, shuffle=True)
    else:
        xs, ys = mnist.test.images, mnist.test.labels
    return {x: xs, y_: ys}


# Set up the NN architecture
layer1 = convolution_layer(x_square, [3, 3, 1, 16])
layer2 = pooling_layer(layer1, 2)
layer3 = convolution_layer(layer2, [3, 3, 16, 32])
layer4 = pooling_layer(layer3, 2)
layer4_flat = tf.reshape(layer4, [-1, 7 * 7 * 32])
y = fc_layer(layer4_flat, [7 * 7 * 32, 10])

# Losses and backprop components
diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(diff)
train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run model
config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)  # lock to 1 processor,  see
#  <https://stackoverflow.com/questions/41233635/tensorflow-inter-and-intra-op-parallelism-configuration>
sess = tf.Session(config=config)
# sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
with sess:
    sess.run(init)
    for i in range(600):
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict(True))
            #     test_accuracy = accuracy.eval(feed_dict=feed_dict(False))
            print("step %d, train acc.: %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict=feed_dict(True))
    print("%d runs gives test accurracy: %g" % (i + 1, accuracy.eval(feed_dict=feed_dict(False))))

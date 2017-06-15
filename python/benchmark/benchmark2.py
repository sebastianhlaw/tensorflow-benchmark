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
 
 Adding save and restore functionality.
 Remove locking to only 1 processor.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DATA_DIR = "../../data/mnist/"
SAVE_DIR = "../../data/saves/"
LOG_DIR = "../../logs/benchmark2/"
PLOT_DIR = "../../data/plots/"

# Random seed setting for reproducible results
tf.set_random_seed(0)  # This is graph-level seed, needs to be set before graph operations executed.
np.random.seed(0)  # Input data shuffle is contolled by numpy seed.
op_level_seed = None  # No need to set op-level seed if graph-level is set, else weights all initialize identically.

# Dimensions of data
X_SIDE = 28
X_DIM = X_SIDE * X_SIDE
Y_DIM = 10

# Import data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Input placeholders
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, X_DIM], name='x_in')
    y_ = tf.placeholder(tf.float32, [None, Y_DIM], name='y_in')

with tf.name_scope('reshape_square'):
    layer0 = tf.reshape(x, [-1, X_SIDE, X_SIDE, 1], name='layer0')
    print("layer0 shape:", layer0.get_shape())
    tf.summary.image('x_square_image', layer0, 10)


def variable_summaries(var):
    """Attach a load of summaries to a Tensor (for TensorBoard visualisation)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=op_level_seed)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolution_layer(x, shape, name, activation=tf.nn.relu):
    """Create 2D convolution layer, based on shape input, with 1,1 strides.
    Args:
        x: A Tensor
        shape: A shape of [height, width, input channels, output channels]
        name: A string to identify layer within graph
        activation: An activation/non-linearity function object
    Returns:
        A Tensor
    """
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([shape[-1]])
            variable_summaries(biases)
        convolution = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
        preactivation = tf.nn.bias_add(convolution, biases)
        return activation(preactivation)


def pooling_layer(x, side, name, pool=tf.nn.max_pool):
    """Create a pooling layer, with no overlaps.
    Args:
        x: A Tensor
        side: Side length of pooling square
        name: A string to identify layer within graph
        pool: A pooling function
    Returns:
        A Tensor
    """
    with tf.name_scope(name):
        layer = pool(x, ksize=[1, side, side, 1], strides=[1, side, side, 1], padding='SAME')
        # tf.summary.image(name+'_image', layer, 10)
        return layer


def fc_layer(x, shape, name, activation=tf.identity):
    """Create a fully-connected layer, optionally add non-linearity/activation.
    Args:
        x: A tensor
        shape: A shape of dimension [input_length, output_length]
        name: A string to identify layer within graph
        activation: An activation/non-linearity function object, including tf.identity
    Returns:
        A Tensor
    """
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(shape)
        with tf.name_scope('biases'):
            biases = bias_variable([shape[1]])
        preactivation = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return activation(preactivation)


def feed_dict(train, batch_size=10):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(batch_size, shuffle=True)
    else:
        xs, ys = mnist.test.images, mnist.test.labels
    return {x: xs, y_: ys}


# def view_kernels(layer):
#
#
# def view_images(layer, image):
#

# Set up the NN architecture
# TODO: dimensions should be flexible, not hardcoded
layer1 = convolution_layer(layer0, [3, 3, 1, 16], 'conv1')
print("layer1 (conv) shape:", layer1.get_shape())
layer2 = pooling_layer(layer1, 2, 'pool1')
print("layer2 (pool) shape:", layer2.get_shape())
layer3 = convolution_layer(layer2, [3, 3, 16, 32], 'conv2')
print("layer3 (conv) hape:", layer3.get_shape())
layer4 = pooling_layer(layer3, 2, 'pool2')
print("layer4 (pool) shape:", layer4.get_shape())
with tf.name_scope('reshape_flatten'):
    layer4_flat = tf.reshape(layer4, [-1, 7 * 7 * 32])
print("layer4flat shape:", layer4_flat.get_shape())
y = fc_layer(layer4_flat, [7 * 7 * 32, 10], 'fc1')

# Losses and backprop components
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Run model
sess = tf.InteractiveSession()
saver = tf.train.Saver()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
# test_writer = tf.summary.FileWriter(LOG_DIR+'/test')

sess.run(tf.global_variables_initializer())
# saver.restore(sess, SAVE_DIR+"model.ckpt")
for i in range(600):
    # if i % 10 == 9:
    #     train_accuracy = accuracy.eval(feed_dict=feed_dict(True))
    #     #     test_accuracy = accuracy.eval(feed_dict=feed_dict(False))
    #     print("step %d, train acc.: %g" % (i, train_accuracy))

    if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        acc = accuracy.eval(feed_dict=feed_dict(False))
        print("test accuracy:", acc)
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()

# save_path = saver.save(sess, SAVE_DIR+"model.ckpt")

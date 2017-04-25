# MNIST for ML beginners
# https://www.tensorflow.org/get_started/mnist/beginners

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set up the model
x = tf.placeholder(tf.float32, [None, 784])  # None indicates dimension can be any length
W = tf.Variable(tf.zeros([784, 10]))  # weights
b = tf.Variable(tf.zeros([10]))  # biases
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))       # error function (naughty)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # error function (nice)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the model performance
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

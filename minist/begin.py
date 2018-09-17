#coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# input raw data
x = tf.placeholder("float", [None, 784])

# parameter to learn
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# output lable
y = tf.nn.softmax(tf.matmul(x,w) + b)

# correct lable
y_ = tf.placeholder("float", [None, 10])

# loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

minist = input_data.read_data_sets("/Users/iig-apple-3/Documents/Data/minist/", one_hot=True)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_yx = minist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_yx})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: minist.test.images, y_: minist.test.labels})   
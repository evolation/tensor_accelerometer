import tensorflow as tf
import numpy as np
# find the
# print(tf.__version__)
# tf.compat.v1.disable_eager_execution()
#
# x = tf.compat.v1.placeholder(tf.float32)
# y = tf.compat.v1.placeholder(tf.float32)
# b = tf.Variable([0.0], name="b")
# ys = tf.multiply(tf.multiply(x, x), b)
#
# e = tf.square(ys - y)
# train = tf.compat.v1.train.GradientDescentOptimizer(1.0).minimize(e)
#
# model = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as session:
#     session.run(model)
#     for i in range(50):
#         x_value = np.random.rand()
#         y_value = 4.0 * x_value * x_value
#         session.run(train, feed_dict={x: x_value, y: y_value})
#
#     b_value = session.run(b)
#     print("predicted model: b = ", b_value)

tf.compat.v1.disable_eager_execution()

#input
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
# output
Y = [[0],
     [1],
     [1],
     [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[4, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[4, 1])

W1 = tf.compat.v1.Variable([[0.6, 0.5], [0.4, 0.1]], shape=[2, 2])
W2 = tf.compat.v1.Variable([[0.1], [0.2]], shape=[2, 1])

B1 = tf.compat.v1.Variable([0.0, 0.0], shape=[2])
B2 = tf.compat.v1.Variable([0.0], shape=1)

# Hidden layer and outout layer
output = tf.compat.v1.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(x, W1) + B1), W2) + B2)

# error estimation
e = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(y, output))
train = tf.compat.v1.train.AdamOptimizer(0.1).minimize(e)

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

for i in range(100001):
    error = sess.run(train, feed_dict={x: X, y: Y})
    if i % 10000 == 0:
        print('\nEpoch: ' + str(i))
        print('\nError: ' + str(sess.run(e, feed_dict={x: X, y: Y})))
        for el in sess.run(output, feed_dict={x: X, y: Y}):
            print('    ', el)
sess.close()

print("Complete")


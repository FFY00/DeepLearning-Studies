#       Tensorflow #1 Example
#   Tensorflow example of Gradient Descent
#   on a linear equation (y = mx + b) with
#   a Plot showing the values learning curve
#
#   https://github.com/FFY00/DeepLearning-Studies

import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m * x + b # y = mx + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) # Also known as r^2
loss = tf.reduce_sum(squared_deltas)

# If you decrease the learning rate, you have to increase the loop range value
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Create Plot
fig = plt.figure(1, figsize=(16, 12))
ax = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax.set_xlabel('m (value)')
ax.set_ylabel('b (value)')
ax.set_zlabel('loss (value)')
ax2.set_xlabel('m (value)')
ax2.set_ylabel('b (value)')
ax3.set_xlabel('m (value)')
ax3.set_ylabel('loss (value)')
ax4.set_xlabel('b (value)')
ax4.set_ylabel('loss (value)')

x_set = [1, 2, 3, 4]
y_set = [0, -1, -2, -3]

for i in range(1000):
    sess.run(train, {x: x_set, y: y_set})
    m_value_plt, b_value_plt, loss_plt = sess.run([m, b, loss], {x: x_set, y: y_set})
    ax.scatter(m_value_plt, b_value_plt, loss_plt)
    ax2.scatter(m_value_plt, b_value_plt)
    ax3.scatter(m_value_plt, loss_plt)
    ax4.scatter(b_value_plt, loss_plt)

m_value, b_value, loss = sess.run([m, b, loss], {x: x_set, y: y_set})
print "y = {}x + {}".format(repr(m_value[0]), repr(b_value[0]))
print "Loss: ", loss

plt.show()
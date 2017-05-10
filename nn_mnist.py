import gzip
import cPickle

import tensorflow as tf
import numpy as np

# Authors: Miriam Cabrera & Geraldo Rodrigues 

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
x_data_train, y_data_train = train_set

y_data_train = one_hot(y_data_train.astype(int),10)

x_data_valid, y_data_valid = valid_set

y_data_valid = one_hot(y_data_valid.astype(int),10)
x_data_test, y_data_test = test_set

y_data_test = one_hot(y_data_test.astype(int),10)


# ---------------- Visualizing some element of the MNIST dataset --------------

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
#
# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]


# TODO: the neural net!!

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Coge el conjunto de datos enteros
print "----------------------"
print "   Start validation...  "
print "----------------------"

batch_size = 20

# for epoch in xrange(100):
#     for jj in xrange(len(x_data) / batch_size):
#         batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
#         batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
#         sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
#
#     print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
#     result = sess.run(y, feed_dict={x: batch_xs})
#     for b, r in zip(batch_ys, result):
#         print b, "-->", r
#     print "----------------------------------------------------------------------------------"
#
# print "----------------------"
# print "   Start training...  "
# print "----------------------"
#
# batch_size = 20

error_valid = 10000.0
error_ant = 20000
epoch = 0;

while error_ant > error_valid:
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    epoch = epoch + 1
    error_ant = error_valid
    error_valid = sess.run(loss, feed_dict={x: x_data_valid, y_: y_data_valid})
    print "Epoch: ", epoch, "Error: ", error_valid

    print "----------------------------------------------------------------------------------"

    # result = sess.run(y, feed_dict={x: batch_xs})
    # for b, r in zip(batch_ys, result):
    #      print b, "-->", r
    #  print "----------------------------------------------------------------------------------"


print "----------------------"
print "   Start test...  "
print "----------------------"

batch_size = 20

# for jj in xrange(len(x_data_test) / batch_size):
#     batch_xs = x_data_test[jj * batch_size: jj * batch_size + batch_size]
#     batch_ys = y_data_test[jj * batch_size: jj * batch_size + batch_size]
        #sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})


bien= 0
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    if np.argmax(b) != np.argmax(r):
        bien = bien + 1
    print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "Clasifico mal: ",bien

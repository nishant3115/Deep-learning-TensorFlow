" Script for fashion mnist classification task using tensorflow"

# load packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data # load input_data function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# download fmnist data and load data
data = input_data.read_data_sets('C:/Users/nverma/Desktop/fmnist_data/', one_hot=True)

# extract train and test sets
train_data = data.train.images
train_labels = data.train.labels
test_data = data.test.images
test_labels = data.test.labels

# inspect shape of data
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# dictionary of labels
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
 }

# look at some examples
plt.figure(figsize=[5, 5])
plt.subplot(1, 2, 1)
curr_img = np.reshape(train_data[0], (28, 28))
curr_lbl = label_dict.get(np.argmax(train_labels))
plt.imshow(curr_img, 'gray')
plt.title(curr_lbl)

plt.subplot(1, 2, 2)
curr_img = np.reshape(test_data[0], (28, 28))
curr_lbl = label_dict.get(np.argmax(test_labels))
plt.imshow(curr_img, 'gray')
plt.title(curr_lbl)

# reshape train & test data in 2D image form
train_X = train_data.reshape(-1, 28, 28, 1)
train_Y = train_labels

test_X = test_data.reshape(-1, 28, 28, 1)
test_Y = test_labels

#################################
# define the tensorFlow graph
################################
n_inputs = train_X.shape[1]
n_classes = train_Y.shape[1]

# input & output placeholders
x = tf.placeholder("float32", [None, n_inputs, n_inputs, 1])
y = tf.placeholder("float32", [None, n_classes])


# define convolution & max pooling layers
def conv_2d_layer(x, W, b, strides, padding):
    """ 
    x : input 4D tensor of size [batch, n_inputs, n_inputs, n_channels]
    W : filter 4D tensor of size [filter_height, filter_width, n_channels, out_channels]
    strides: 1D tensor of length 4 (stride length for each dimension of input x)
    padding: either "SAME' or "VALID"
    """
    x = tf.nn.conv2d(x, W, strides, padding)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x


def max_pool_layer(x, kernel_size, strides, padding):
    """
    x : input 4D tensor of size [batch, n_inputs, n_inputs, n_channels]
    kernel_size : 1D tensor of length 4 (window size for each dimension of input x)
    strides: 1D tensor of length 4 (stride length for each dimension of input x)
    padding: either "SAME' or "VALID"
    """
    x = tf.nn.max_pool(x, kernel_size, strides, padding)
    return x


# define parameter dictionaries
weights = {
        'wc0': tf.get_variable(name='W0', shape=[3, 3, 1, 32], dtype='float32',
                               initializer=tf.contrib.layers.xavier_initializer()),
        'wc1': tf.get_variable(name='W1', shape=[3, 3, 32, 64], dtype='float32',
                               initializer=tf.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable(name='W2', shape=[3, 3, 64, 128], dtype='float32',
                               initializer=tf.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable(name='W3', shape=[4*4*128, 128], dtype='float32',
                               initializer=tf.contrib.layers.xavier_initializer()),
        'wc4': tf.get_variable(name='W4', shape=[128, n_classes], dtype='float32',
                               initializer=tf.contrib.layers.xavier_initializer())
        }

biases = {
        'b0': tf.get_variable(name='B0', shape=[32], dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer()),
        'b1': tf.get_variable(name='B1', shape=[64], dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer()),
        'b2': tf.get_variable(name='B2', shape=[128], dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer()),
        'b3': tf.get_variable(name='B3', shape=[128], dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer()),
        'b4': tf.get_variable(name='B4', shape=[n_classes], dtype='float32',
                              initializer=tf.contrib.layers.xavier_initializer()),
        }


# define a small CNN network graph
def conv_net(x, weights, biases):
    # 1st layer: convolutional layer (32 3x3 filters)
    conv1 = conv_2d_layer(x, weights['wc0'], biases['b0'], [1, 1, 1, 1], 'SAME')
    # 2nd layer: max pooling (2x2 filter max pooling)
    max_pool1 = max_pool_layer(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    # 3rd layer: convolutional layer (64 3x3 layers)
    conv2 = conv_2d_layer(max_pool1, weights['wc1'], biases['b1'], [1,1,1,1], 'SAME')
    # 4th layer: max pooling (2x2 filter max pooling)
    max_pool2 = max_pool_layer(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    # 5th layer: convolutional layer (128 3x3 filters)
    conv3 = conv_2d_layer(max_pool2, weights['wc2'], biases['b2'], [1, 1, 1, 1], 'SAME')
    # 6th layer: max pooling (2x2 filter max pooling)
    max_pool3 = max_pool_layer(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    # 7th layer: fully connected layer (128 nodes)
    dense = tf.reshape(max_pool3, [-1, weights['wc3'].get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, weights['wc3']), biases['b3']))
    # 8th layer: fully connected layer (n_classes nodes)
    out = tf.add(tf.matmul(dense,weights['wc4']), biases['b4'])

    return out


# define loss and optimizer functions for model training
sample_prediction = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sample_prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# define accuracy metric
correct_prediction = tf.equal(tf.argmax(sample_prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# training params
training_iters = 200
batch_size = 128

# start tensorFlow session
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    # training epochs
    for i in range(training_iters):
        # batch size (# of training runs/epoch)
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size, len(train_X))]
            batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size, len(train_Y))]
            # Run optimization op (backprop)
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss and accuracy
            (loss, acc) = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        (test_acc, valid_loss) = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_Y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    summary_writer.close()

# plot results
plt.figure()
plt.plot(range(training_iters), train_loss, 'b', label='Training loss')
plt.plot(range(training_iters), test_loss, 'r', label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')

plt.figure()
plt.plot(range(training_iters), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(training_iters), test_accuracy, 'r', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')

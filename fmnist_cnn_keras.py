" Script for fashion mnist classification task using tensorflow and keras API"

# load packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# import mnist fashion data
fashion_mnist = keras.datasets.fashion_mnist

# load train, test images and labels (60000 training, 10000 test)
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
print(train_images.shape)  # 60k images of 28x28 size
print(len(np.unique(train_labels)))  # 10 unique classes
print(test_images.shape)  # 10k images of 28x28 size

# define classes with names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data visualization & preprocessing
plt.figure()
plt.imshow(train_images[231], 'gray')
plt.colorbar()
plt.grid(False)

# normalize images b/w 0 and 1
print(train_images.dtype)
train_images = train_images/255
print(train_images.dtype)  # data type automatically converted from uint8 to float64
test_images = test_images/255

# display images (subplot type plotting)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], 'gray')
    plt.xlabel(class_names[train_labels[i]])

# define neural net model
nn_model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                             keras.layers.Dense(128, activation=tf.nn.relu),
                             keras.layers.Dense(10, activation=tf.nn.softmax)])
     
# compile the neural network model     
nn_model.compile(optimizer=tf.train.AdamOptimizer(),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

# train model
history = nn_model.fit(train_images, train_labels, epochs=50)

# plot training results
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

# PLOT loss over epochs
plt.figure()
plt.plot(epochs, loss, 'b')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# PLOT accuracy over epochs
plt.figure()
plt.plot(epochs, acc, 'r')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# performance of neural network on test set
(test_loss, test_acc) = nn_model.evaluate(test_images, test_labels)
print('Test accuracy is:', test_acc)
print('Test loss is:', test_loss)

# predict class probabilities on test cases
test_predictions = nn_model.predict(test_images)

# class with highest probabilities for each test image
test_classes = np.argmax(test_predictions, axis=1)





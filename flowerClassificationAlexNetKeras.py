import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflowerData
from alexnet_model import alexnet
import matplotlib.pyplot as plt

# Download data
(data, labels) = oxflowerData.load_data(one_hot=True)

# import Alexnet model
nn_model = alexnet(data.shape[1:], labels.shape[1])

# Compile model
nn_model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model (20% data used for validation)
history = nn_model.fit(data, labels, batch_size=64, epochs=100, verbose=1, validation_split=0.2, shuffle=True)

# plot training & validation results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# PLOT LOSS over epochs
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# PLOT accuracy over epochs
plt.figure()
plt.plot(epochs, acc, 'ro', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
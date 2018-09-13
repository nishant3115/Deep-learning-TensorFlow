from tensorflow import keras


def alexnet():
    # Sequential model
    model = keras.Sequential()

    # 1st Convolutional Layer
    model.add(keras.layers.Conv2D(filters=96, input_shape = (224,224,3), kernel_size=(11, 11),
                                  strides=(4, 4), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(keras.layers.BatchNormalization())

    # 2nd Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # 3rd Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # 4th Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # 5th Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # Passing it to a dense layer
    model.add(keras.layers.Flatten())
    # 1st Dense Layer
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # 2nd Dense Layer
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # 3rd Dense Layer
    model.add(keras.layers.Dense(1000))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(keras.layers.BatchNormalization())

    # Output Layer
    model.add(keras.layers.Dense(17))
    model.add(keras.layers.Activation('softmax'))

    return model
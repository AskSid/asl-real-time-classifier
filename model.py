import pandas as pd
import os
import cv2 as cv
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt


def load_data(data_path):
    images = []
    labels = []

    letterDirectories = os.listdir(data_path)
    i = 0
    totalImages = 0
    for letterDirectory in letterDirectories[1:]:
        print(letterDirectory)
        numLetterImages = 0
        letterImages = os.listdir(data_path + "/" + letterDirectory)
        for letterImage in letterImages:
            if letterImage.endswith(".jpg") or letterImage.endswith(".jpeg"):
                numLetterImages += 1
                image = cv.imread(data_path + "/" + letterDirectory + "/" + letterImage)
                images.append(cv.resize(image, (128, 128)))
                labels.append(i)
        print("Loaded in " + numLetterImages + " images for " + letterDirectory)
        total += numLetterImages
        i += 1
    print("Total images in dataset: " + totalImages)
    return images, labels

def preprocess(images, labels):
    images = np.array(images)
    images = images.astype('float32') / 255.0
    labels = utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1)
    print("Preprocessed data.")
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(24, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_split=0.2, shuffle = True, verbose=1)
    model.save('modelAugmented.h5')


    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


images, labels = load_data("../../../../Downloads/augmented")
print()
x_train, x_test, y_train, y_test = preprocess(images, labels)
print()
train_model(x_train, y_train, x_test, y_test)


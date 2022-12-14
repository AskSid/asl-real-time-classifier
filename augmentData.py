from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
import os

datagen = ImageDataGenerator(
        horizontal_flip=True)

data_path = "../../../../Downloads/liveDataset"
letterDirectories = os.listdir(data_path)
for letterDirectory in letterDirectories[1:]:
    print(letterDirectory)
    letterImages = os.listdir(data_path + "/" + letterDirectory)
    for letterImage in letterImages:
        if letterImage.endswith(".jpg"):
            image = cv.imread(data_path + "/" + letterDirectory + "/" + letterImage)
            img = np.array(image)
            x = img.reshape((1,) + img.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                          save_to_dir="../../../../Downloads/augmented/" + letterDirectory, save_prefix='asl', save_format='jpg'):
                i += 1
                if i > 1:
                    break  # otherwise the generator would loop indefinitely
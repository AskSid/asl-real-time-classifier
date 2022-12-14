This repository contains the files necessary to generate your own dataset with hand movements, augment data with different features, train a model with that dataset, and finally a program to actually teach ASL using that network

generateDataLivefeed.py starts a video and begins to save the video data of your hand to a folder. It overlays mediapipe hand landmarks and saves only the bounding box. The user can set a filepath to save the data in a specific directory. This was used to generate our dataset of ASL. 

augmentData.py then takes an existing dataset, with folders of each letter, and augments that data using ImageDataGenerator from Keras. We chose to just use a horizontal flip to account for left and right hand signers as our own created dataset only uses right handed signs. This can be modified to include data rotations, zooms, flips, and translations but this may reduce accuracy betweeen certain letters such as P and K.

model.py contains the architecture used to train the images in order to classify them into letters A-Z excluding J and Z since they require motion to sign. This currently runs with 99.6 percent accuracy on our dataset with 3 epochs.

main.py uses the architecture saved from model.py to run a program to teach ASL. It randomly generates a letter on the screen with text. It then asks the user to sign accordingly. If they produce the correct sign based on the pre-trained model, it outputs "CORRECT ANSWER" otherwise it outputs "INCORRECT ANSWER." 

In order to run our code, simply run python main.py and the program will run using weights from modelRaw.h5

(modelRaw.h5 is trained without any augmentation while modelAugment.h5 is trained with horizontal flipping)

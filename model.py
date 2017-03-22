import csv
import cv2
import numpy as np

DATA_PATH = './data/'
EPOCHS = 5
BATCH_SIZE = 64

samples = []
with open(DATA_PATH + 'driving_log.csv', 'r') as csv_file:
    for row in csv.reader(csv_file):
        samples.append(row)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                fliped_img = cv2.flip(center_image, 1)
                fliped_angle = -center_angle
                images.append(fliped_img)
                angles.append(fliped_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

row, col, ch = 160, 320, 3  # image format
print('data loaded')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch)))
# using NVidia network
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
 steps_per_epoch=len(train_samples)/BATCH_SIZE,
  validation_data=validation_generator,
  validation_steps=len(validation_samples)/BATCH_SIZE,
  epochs=EPOCHS)

print(history_object.history.keys())

model.save('model.h5')

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_epoch.png')

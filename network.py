import csv
import cv2
import numpy as np

DATA_PATH = './data/'

images = []
measurements = []
with open(DATA_PATH + 'driving_log.csv') as csv_file:
    for row in csv.reader(csv_file):
        img_center = cv2.imread(row[0])
        img_left = cv2.imread(row[1])
        img_right = cv2.imread(row[2])
        images.extend((img_center, img_left, img_right))

        steering_center = float(row[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        measurements.extend((steering_center, steering_left, steering_right))

def double_data():
    for index in range(0, len(images), 3):
        imf_center = np.fliplr(images[index])
        imf_left = np.fliplr(images[index+1])
        imf_right = np.fliplr(images[index+2])
        images.extend((imf_center, imf_left, imf_right))

    for index in range(0, len(measurements), 3):
        sf_center = -measurements[index]
        sf_left = -measurements[index+1]
        sf_right = -measurements[index+2]
        measurements.extend((sf_center, sf_left, sf_right))

double_data()

X_train = np.array(images)
y_train = np.array(measurements)

print('data loaded')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 5.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=5, verbose=1)

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
plt.show()

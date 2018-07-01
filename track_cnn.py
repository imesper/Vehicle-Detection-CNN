import sklearn
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers, utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Activation, Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout, LeakyReLU, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle, randint
import numpy as np
import cv2
import csv
import os

# Load data
# '../DrivingData02/


def loadData():
    # Read in cars and notcars
    samples = []
    labels = []
    for file in os.listdir('./vehicles'):
        if os.path.isdir('./vehicles/'+file):
            for filename in os.listdir('./vehicles/'+file):
                if '.png' in filename:
                    samples.append('./vehicles/'+file+'/'+filename)
                    labels.append(1)

    for filename in os.listdir('./aug_vehicles'):
        if '.png' in filename:
            samples.append('./aug_vehicles/'+filename)
            labels.append(1)

    for file in os.listdir('./non-vehicles'):
        if os.path.isdir('./non-vehicles/'+file):
            for filename in os.listdir('./non-vehicles/'+file):
                if '.png' in filename:
                    samples.append('./non-vehicles/'+file+'/'+filename)
                    labels.append(0)

    for filename in os.listdir('./aug_non-vehicles'):
        if '.png' in filename:
            samples.append('./aug_non-vehicles/'+filename)
            labels.append(0)

    samples, labels = sklearn.utils.shuffle(samples, labels)
    return samples, labels


def randomDarkener(image):
    alpha = 1
    beta = randint(-30, 0)
    res = cv2.addWeighted(image, alpha, np.zeros(
        image.shape, image.dtype), 0, beta)
    return res


def getData(samples, labels):
    num_samples = len(samples)
    samples, labels = sklearn.utils.shuffle(samples, labels)
    images = []
    for sample in samples:
        if os.path.isfile(sample):
            image = cv2.cvtColor(cv2.imread(
                sample), cv2.COLOR_BGR2RGB)
            images.append(image)
        else:
            exit(-1)

    # trim image to only see section with road
    X_train = np.array(images)
    y_train = np.array(labels)
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
    # y_train = utils.to_categorical(y_train, num_classes=2)
    return X_train, y_train


def generator(samples, labels, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples, labels = sklearn.utils.shuffle(samples, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            images = []
            for i, batch_sample in enumerate(batch_samples):
                if os.path.isfile(batch_sample):
                    # print(batch_sample, batch_labels[i])
                    image = cv2.cvtColor(cv2.imread(
                        batch_sample), cv2.COLOR_BGR2RGB)
                    images.append(image)
                else:
                    exit(-1)

        # trim image to only see section with road
        X_train = np.array(images)
        y_train = np.array(batch_labels)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        # y_train = utils.to_categorical(y_train, num_classes=2)
        yield X_train, y_train


def createModel():

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(32, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def CNN():

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))
    model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()

    return model


def main():
    # image = cv2.cvtColor(cv2.imread(
    #     './test_images/test1.jpg'), cv2.COLOR_BGR2GRAY)
    # print(image.reshape(
    #     1, 720, 1280, 1))
    # exit(-1)
    # Loading Data from 3 different folders
    # Each folder has different runs on simulator
    samples, labels = loadData()

    print(len(samples), len(labels))

    # Spliting the data between trainnig (80%) and validation (20%)
    train_samples, validation_samples, train_labels, validation_labels = train_test_split(
        samples, labels, test_size=0.2)

    X_train, y_train = getData(train_samples, train_labels)
    X_val, y_val = getData(validation_samples, validation_labels)

    datagen = ImageDataGenerator()

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    valgen = ImageDataGenerator()
    valgen.fit(X_val)

    # Setting the batch size
    batch_size = 32
    # Getting the model
    model = CNN()
    # model = createModel()
    # Running the model, saving only the best models based on validation loss
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

    train_generator = generator(
        train_samples, train_labels, batch_size=batch_size)
    validation_generator = generator(
        validation_samples, validation_labels, batch_size=batch_size)

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size, validation_data=valgen.flow(X_val, y_val, batch_size=batch_size),
                        validation_steps=len(validation_samples)/batch_size, epochs=30, callbacks=[checkpoint])


if __name__ == '__main__':
    main()

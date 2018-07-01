
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot


def loadData(path, label):
    # Read in cars and notcars
    samples = []
    labels = []
    for file in os.listdir(path):
        if os.path.isdir(path+file):
            for filename in os.listdir(path+file):
                if '.png' in filename:
                    samples.append(mpimg.imread(path+file+'/'+filename))
                    labels.append(label)
    return samples, labels


X_car, y_car = loadData('./vehicles/', 1)
X_noncar, y_noncar = loadData('./non-vehicles/', 0)
X_car = np.array(X_car)
y_car = np.array(y_car)
X_noncar = np.array(X_noncar)
y_noncar = np.array(y_noncar)
print(len(X_car))
noncargen = ImageDataGenerator(horizontal_flip=True,
                               rotation_range=20,
                               width_shift_range=.2,
                               height_shift_range=.2,
                               fill_mode='nearest'
                               )
cargen = ImageDataGenerator(horizontal_flip=True,
                            rotation_range=20,
                            width_shift_range=.2,
                            height_shift_range=.2,
                            fill_mode='nearest'
                            )

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
cargen.fit(X_car)
noncargen.fit(X_noncar)
batches = 0
for X_batch, y_batch in cargen.flow(X_car, y_car, batch_size=32, save_to_dir='aug_vehicles', save_prefix='aug', save_format='png'):
    batches += 1
    if batches >= len(X_car) / 32:
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break

batches = 0
for X_batch, y_batch in noncargen.flow(X_noncar, y_noncar, batch_size=32, save_to_dir='aug_non-vehicles', save_prefix='aug', save_format='png'):
    batches += 1
    if batches >= len(X_noncar) / 32:
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break

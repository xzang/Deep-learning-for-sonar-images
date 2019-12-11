# -*- coding: utf-8 -*-
"""
Written by Tim Yin and Xiaoqin Zang

Last editted by Xiaoqin Zang on Nov. 8, 2019

"""

from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import imageio
import glob
import numpy as np


from sklearn.model_selection import RepeatedKFold


######## read eel data
eel_array = np.empty((0, 104, 104))

for im_path in glob.glob("C:/Users/****/Documents/sonar_camera/coding/eel/eel_diff_wave/*.png"):
     result = np.zeros((104, 104))
     im = imageio.imread(im_path)
     result[22:(im.shape[0]+22),22:(im.shape[1]+22)] = im
     eel_array = np.append(eel_array, [result], axis=0)

#eel_array.shape
     ######### downsample
eel_array = eel_array[0::7]

########## read short stick data
stick_array_short = np.empty((0, 104, 104))

for im_path in glob.glob("C:/Users/****/Documents/sonar_camera/coding/sti_diff_wave_short/*.png"):
     result = np.zeros((104, 104))
     im = imageio.imread(im_path)
     result[22:(im.shape[0]+22),22:(im.shape[1]+22)] = im
     stick_array_short = np.append(stick_array_short, [result], axis=0)

#stick_array_short.shape
stick_array_short = stick_array_short[0::7]



stick_array = stick_array_short
eel_size = eel_array.shape[0]
stick_size = stick_array.shape[0]
all_array = np.concatenate((eel_array, stick_array), axis=0)
y = np.repeat(np.array([1, 0], dtype=np.int64), [eel_size, stick_size], axis=0)


##### shuffle the dataset
shuffle_ix_1 = list(range(len(all_array)))

np.random.seed(567)

np.random.shuffle(shuffle_ix_1)
all_array = all_array[shuffle_ix_1]
y = y[shuffle_ix_1]



seed = 987
np.random.seed(seed)

# ten-fold cross validation
kfold       = RepeatedKFold(n_splits=10, n_repeats=5, random_state=seed)
cvscores    = []
batch_size  = 32 #64 if the dataset is bigger
num_classes = 1
epochs      = 5

# input image dimensions
img_rows, img_cols = 104, 104

for train, test in kfold.split(all_array, y):
    x_train = all_array[train]
    y_train = y[train]
    x_test = all_array[test]
    y_test = y[test]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape, padding='valid'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 0,
          shuffle = 1,
          #validation_data=(x_test, y_test)
          )
    
    scores = model.evaluate(x_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))





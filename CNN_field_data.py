# -*- coding: utf-8 -*-
"""
Written by Tim Yin and Xiaoqin Zang

Last editted by Xiaoqin Zang on Nov. 8, 2019
"""


from __future__ import print_function
import keras

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D #, BatchNormalization, Activation
from keras import backend as K

import imageio
import glob
import numpy as np
from random import randrange 


#from sklearn.metrics import confusion_matrix
#from collections import Counter

import pandas as pd

import csv




def get_field_data(root_path, group):
    field_data = []
    folder_name = [name for name in os.listdir(root_path) if name.startswith(group)]
    new_dir = os.path.join(root_path, folder_name[0])
    
    object_folder_name = [name for name in os.listdir(new_dir) if os.path.isdir(os.path.join(
            new_dir, name))]
        
    for i in range(len(object_folder_name)):
        object_dir = os.path.join(new_dir, object_folder_name[i])
        
        object_array = np.empty((0, 61, 61))
        for im_path in glob.glob(object_dir + "/*.png"):
            if "diffwvlt" in im_path:
                im = imageio.imread(im_path)
                object_array = np.append(object_array, [im], axis=0)
        field_data.append(object_array)
    return object_folder_name, field_data



def get_train_test_data(eel_data, noneel_data, percentage_1, percentage_2, seed):
    
    shuffle_ix_eel = list(range(len(eel_data)))
    np.random.seed(seed)
    np.random.shuffle(shuffle_ix_eel)
    eel_data_train = [eel_data[i] for i in shuffle_ix_eel[:round(len(shuffle_ix_eel)*percentage_1)]]
    eel_data_test = [eel_data[i] for i in shuffle_ix_eel[round(len(shuffle_ix_eel)*percentage_1):]]
    
    eel_data_test_video = shuffle_ix_eel[round(len(shuffle_ix_eel)*percentage_1):]
    
    shuffle_ix_noneel = list(range(len(noneel_data)))
    np.random.seed(seed+randrange(1000))
    np.random.shuffle(shuffle_ix_noneel)
    noneel_data_train = [noneel_data[i] for i in shuffle_ix_noneel[:round(len(shuffle_ix_noneel)*percentage_2)]]
    noneel_data_test = [noneel_data[i] for i in shuffle_ix_noneel[round(len(shuffle_ix_noneel)*percentage_2):]]
    
    noneel_data_test_video = shuffle_ix_noneel[round(len(shuffle_ix_noneel)*percentage_2):]
    
    train_eel_data = np.concatenate([eel_data_train[i] for i in range(len(eel_data_train))], axis=0)
    train_noneel_data = np.concatenate([noneel_data_train[i] for i in range(len(noneel_data_train))], axis=0)
    
    test_eel_data = np.concatenate([eel_data_test[i] for i in range(len(eel_data_test))], axis=0)
    test_noneel_data = np.concatenate([noneel_data_test[i] for i in range(len(noneel_data_test))], axis=0)
    
    train_data = np.concatenate((train_eel_data, train_noneel_data), axis=0)
    test_data = np.concatenate((test_eel_data, test_noneel_data), axis=0)
    
    test_eel_data_size = []
    for eel in eel_data_test:
        test_eel_data_size.append(eel.shape[0])
        
    test_noneel_data_size = []
    for noneel in noneel_data_test:
        test_noneel_data_size.append(noneel.shape[0])
    
    train_label = np.repeat(np.array([1, 0], dtype=np.int64), [len(train_eel_data), len(train_noneel_data)], axis=0)
    test_label = np.repeat(np.array([1, 0], dtype=np.int64), [len(test_eel_data), len(test_noneel_data)], axis=0)
    
    shuffle_ix_train = list(range(len(train_data)))
    np.random.seed(seed+randrange(1000))
    np.random.shuffle(shuffle_ix_train)
    
    train_data = train_data[shuffle_ix_train]
    train_label = train_label[shuffle_ix_train]
    
    return train_data, train_label, test_data, test_label, eel_data_test_video, test_eel_data_size, \
            noneel_data_test_video, test_noneel_data_size




def train_test_model(x_train, y_train, x_test, y_test, test_eel_data_size, test_noneel_data_size, num_epo):
    batch_size = 32
    num_classes = 1
    epochs = num_epo
    img_rows, img_cols = 61, 61
    
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
    
    for i in range(len(x_train)):
        x_train[i, :, :, 0] /= x_train[i, :, :, 0].max()
    
    for i in range(len(x_test)):
        x_test[i, :, :, 0] /= x_test[i, :, :, 0].max()
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
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
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, \
                                                  amsgrad=False),
                  metrics=['binary_accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs = epochs,
              verbose = 1,
              shuffle = 1,
              validation_data=(x_test, y_test), 
              class_weight = {0: 0.6, 1: 0.4}
              )
    
    y_pred = model.predict_classes(x_test, batch_size=None, verbose=1)
    
    eel_result = [sum(y_pred[:test_eel_data_size[0]])/test_eel_data_size[0]]
    for i in range(1, len(test_eel_data_size)):
        temp_result = y_pred[range(sum(test_eel_data_size[:i]), sum(test_eel_data_size[:(i+1)]))]
        eel_result.append(sum(temp_result)/len(temp_result))
        
    y_pred_noneel = y_pred[sum(test_eel_data_size):]
    
    noneel_result = [1-sum(y_pred_noneel[:test_noneel_data_size[0]])/test_noneel_data_size[0]]
    for i in range(1, len(test_noneel_data_size)):
        temp_result = y_pred_noneel[range(sum(test_noneel_data_size[:i]), sum(test_noneel_data_size[:(i+1)]))]
        noneel_result.append(1-sum(temp_result)/len(temp_result))
    
#    result = [sum(i>threshold for i in eel_result)[0], sum(i>=(1-threshold) for i in noneel_result)[0]]
    
    ### output the accuracy of each test eel and test noneel
    eel_output = []
    noneel_output = []
    for i in range(9):
        threshold = (i+1)/10
        eel_output = np.append(eel_output,[sum(i>threshold for i in eel_result)[0]],axis=0)  
        noneel_output = np.append(noneel_output,[sum(i>=(1-threshold) for i in noneel_result)[0]],axis=0)
    
    return eel_output.reshape(1,9), noneel_output.reshape(1,9)



##### main()
    
# import field images
object_folder_name_eel, eel_data = get_field_data("C:/Users/****/Documents/sonar_camera/coding/field/eel/", "12") # eels in Tier 1 and 2
object_folder_name_noneel, noneel_data = get_field_data("C:/Users/****/Documents/sonar_camera/coding/field/noneel/", "stick_pipe") # sticks and pipes

### 50 randomizations
result_eel    = pd.DataFrame(columns=['10%', '20%','30%', '40%','50%', '60%','70%', '80%','90%'])  # percentage threshold
result_noneel = pd.DataFrame(columns=['10%', '20%','30%', '40%','50%', '60%','70%', '80%','90%'])
num_epoch     = 6
num_randm     = 50
perctg_eel    = 0.8  # 80% eel images for training
perctg_noneel = 0.8  # 80% noneel images for training

for j in range(num_randm):  # multiple times of randomization
        
    x_train, y_train, x_test, y_test, eel_data_test_video, test_eel_data_size, noneel_data_test_video, test_noneel_data_size \
        = get_train_test_data(eel_data, noneel_data, perctg_eel, perctg_noneel, j+3000)

    eel_output, noneel_output = train_test_model(x_train, y_train, x_test, y_test, test_eel_data_size, test_noneel_data_size, num_epoch) 
    result_eel.loc[j]    = eel_output[0]/len(test_eel_data_size)
    result_noneel.loc[j] = noneel_output[0]/len(test_noneel_data_size)
    
os.chdir("C:/Users/****/Documents/sonar_camera/coding/")
with open('field_results.csv', 'w', newline='') as eelfile:
    wr = csv.writer(eelfile, quoting=csv.QUOTE_ALL)
    wr.writerow(result_eel.mean())
    wr.writerow(result_noneel.mean())
    wr.writerow(result_eel.std())
    wr.writerow(result_noneel.std())









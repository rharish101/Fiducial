#!/usr/bin/env python
import os
from scipy.ndimage import gaussian_filter
from cv2 import imread
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def extract_data():
    images = []
    labels = []

    folder = '../Fiducial data/PVC skull model/Sequential scan/Patient-BARC '\
            'ACRYLIC SKULL/Study_34144_CT_SKULL[20160627]/Series_002_Plain Scan/'
    for i in range(5):
        for img in os.listdir(folder):
            image = imread(folder + img, 0)
            for _ in range(10):
                a = np.random.uniform(0, image.shape[0] - 50, []).astype(np.int)
                b = np.random.uniform(0, image.shape[0] - 50, []).astype(np.int)
                image = gaussian_filter(image[a:(a + 50), b:(b + 50)], 2)
                if image.max() > 100:
                    break
            if image.max() > 100:
                images.append(image)
                labels.append(0)

    for fold in os.listdir('.'):
        if not os.path.isdir(fold):
            continue
        for img in os.listdir(fold):
            if img[-4:] != '.png':
                continue
            image = imread(fold + '/' + img, 0)
            images.append(image)
            labels.append(1)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def shuffle_data(data, labels):
    combined = np.column_stack((data, labels))
    np.random.shuffle(combined)
    data = combined.T[:-1].T
    labels = combined.T[-1].T
    return data, labels

def split_data(data, labels, test_split=0.3):
    train_data = data[:-int(len(data) * test_split)]
    train_labels = labels[:-int(len(labels) * test_split)]
    test_data = data[-int(len(data) * test_split):]
    test_labels = labels[-int(len(labels) * test_split):]
    return train_data, train_labels, test_data, test_labels

def get_generator(data):
    generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.15,
                                   height_shift_range=0.15, shear_range=1.57,
                                   cval=0, horizontal_flip=True,
                                   vertical_flip=True)
    generator.fit(data)
    return generator

def get_data_gen(**kwargs):
    data, labels = extract_data()
    data, labels = shuffle_data(np.reshape(data, (-1, 50 * 50)), labels)
    data = np.reshape(data, (-1, 50, 50, 1))
    train_data, train_labels, test_data, test_labels = split_data(data, labels,
                                                                  **kwargs)
    return train_data, train_labels, test_data, test_labels,\
           get_generator(train_data)


#!/usr/bin/env python
from __future__ import print_function
import os
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np

def extract_data():
    images = []
    labels = []

    folder = '../Fiducial data/PVC skull model/Sequential scan/Patient-BARC '\
            'ACRYLIC SKULL/Study_34144_CT_SKULL[20160627]/Series_002_Plain Scan/'
    for i in range(5):
        for img in os.listdir(folder):
            image = cv2.imread(folder + img, 0)
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
            image = cv2.imread(fold + '/' + img, 0)
            images.append(image)
            labels.append(1)

    return images, labels

def augment_data(images, labels, rotation_range=90, width_shift_range=0.15,
                 height_shift_range=0.15, shear_range=60,
                 borderMode=cv2.BORDER_REPLICATE, fill_val=0,
                 horizontal_flip=True, vertical_flip=True, **kwargs):
    total_length = len(images)
    for image, label in zip(images[:total_length], labels[:total_length]):
        if len(image.shape) == 3:
            image = np.squeeze(image, -1)
        images_now = [image]
        rows, cols = image.shape

        # Rotation
        for angle in np.arange(15, rotation_range + 1, 15):
            rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            images_now.append(np.uint8(cv2.warpAffine(images_now[0], rot_matrix,
                                                      (cols, rows),
                                                      borderMode=borderMode,
                                                      borderValue=fill_val)))

        # Translation
        length = len(images_now)
        for w_shift in np.arange(0.075, width_shift_range + 0.01, 0.075):
            for h_shift in np.arange(0.075, height_shift_range + 0.01, 0.075):
                shift_matrix = np.float32([[1, 0, cols * w_shift],
                                           [0, 1, rows * h_shift]])
                for img in images_now[:length]:
                    images_now.append(np.uint8(cv2.warpAffine(img, shift_matrix,
                        (cols, rows), borderMode=borderMode,
                        borderValue=fill_val)))

        # Shear
        length = len(images_now)
        pts1 = np.float32([[0, rows / 2], [cols, rows / 2], [cols / 2, 0]])
        for shear_angle in np.arange(20, shear_range + 1, 20):
            pts2 = np.float32([[0, rows / 2], [cols, rows / 2], [cols / 2,
                                (rows / 2) * (1 - np.cos(np.deg2rad(
                                shear_angle)))]])
            shear_matrix = cv2.getAffineTransform(pts1, pts2)
            for img in images_now[:length]:
                images_now.append(np.uint8(cv2.warpAffine(img, shear_matrix,
                    (cols, rows), borderMode=borderMode,
                    borderValue=fill_val)))

        # Flip
        if vertical_flip:
            length = len(images_now)
            for img in images_now[:length]:
                images_now.append(np.uint8(cv2.flip(img, flipCode=0)))
        if horizontal_flip:
            length = len(images_now)
            for img in images_now[:length]:
                images_now.append(np.uint8(cv2.flip(img, flipCode=1)))

        images += images_now[1:]
        labels += (label * np.ones((len(images_now) - 1,))).tolist()

    return images, labels

def shuffle_data(data, labels):
    combined = np.column_stack((data, labels))
    np.random.shuffle(combined)
    data = combined.T[:-1].T
    labels = combined.T[-1].T
    return data, labels

def split_data(data, labels, test_split=0.3, **kwargs):
    train_data = data[:-int(len(data) * test_split)]
    train_labels = labels[:-int(len(labels) * test_split)]
    test_data = data[-int(len(data) * test_split):]
    test_labels = labels[-int(len(labels) * test_split):]
    return train_data, train_labels, test_data, test_labels

def get_data(from_disk=True, **kwargs):
    keys = ('train_data', 'train_labels', 'test_data', 'test_labels')

    if not from_disk or not set(['template_0_' + arr + '.npy'\
    for arr in keys]).issubset(os.listdir('.')):
        data, labels = extract_data()
        data, labels = augment_data(data, labels, **kwargs)
        data, labels = shuffle_data(np.reshape(data, (-1, 50 * 50)), labels)
        data = np.reshape(data, (-1, 50, 50, 1))
        data = dict(zip(keys, split_data(data, labels, **kwargs)))
        print("Saving to disk...")
        for arr in keys:
            np.save('template_0_' + arr + '.npy', data[arr].astype(np.uint8))
        del data

    data = []
    for arr in keys:
        data.append(np.load('template_0_' + arr + '.npy', mmap_mode='r'))
    return data


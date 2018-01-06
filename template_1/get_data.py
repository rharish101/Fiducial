#!/usr/bin/env python
from __future__ import print_function
import os
#from scipy.ndimage import gaussian_filter
#import cv2
import numpy as np
import sys
sys.path.append('..')
from template_0.get_data import extract_data, augment_data, shuffle_data,\
                                split_data

def get_data(from_disk=True, **kwargs):
    keys = ('train_data', 'train_labels', 'test_data', 'test_labels')

    if not from_disk or not set(['template_1_' + arr + '.npy'\
    for arr in keys]).issubset(os.listdir('.')):
        data, labels = extract_data()
        data, labels = augment_data(data, labels, **kwargs)
        data, labels = shuffle_data(np.reshape(data, (-1, 40 * 40)), labels)
        data = np.reshape(data, (-1, 40, 40, 1))
        data = dict(zip(keys, split_data(data, labels, **kwargs)))
        print("Saving to disk...")
        for arr in keys:
            np.save('template_1_' + arr + '.npy', data[arr].astype(np.uint8))
        del data

    data = []
    for arr in keys:
        data.append(np.load('template_1_' + arr + '.npy', mmap_mode='r'))
    return data


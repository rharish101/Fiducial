#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from final_python import god_function
from process import import_dicom
import cv2

# NOTE: Refer to https://en.wikipedia.org/wiki/Anatomical_plane for orientation
# orientation (z, y, x); origin is corner of cube near right chin
dir_axial = './2016.05.26 Glass Scan 1 mm/Glass Scan Axial 1.25 mm/DICOM/PA1/'\
            'ST1/SE2/'
images_axial = []
for img in sorted(os.listdir(dir_axial),
                  key=lambda img_name: int(img_name[2:]))[3:-2]:
    for _ in range(3):
        images_axial.append(import_dicom(dir_axial + img))
images_axial = images_axial[:-1]

# orientation (x, -z, y)
images_sagittal = np.array(list(map(lambda img: gaussian_filter(img, 1),
    np.swapaxes(np.swapaxes(images_axial, 0, 2), 1, 2)[:, ::-1, :])))

# orientation (y, -z, x)
images_coronal = np.array(list(map(lambda img: gaussian_filter(img, 1),
    np.swapaxes(images_axial, 0, 1)[:, ::-1, :])))

images_axial = np.array(list(map(lambda img: gaussian_filter(img, 1),
                                 images_axial)))

if __name__ == '__main__':
    print(god_function(images_axial, images_coronal, images_sagittal))

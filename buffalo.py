#!/usr/bin/env python
from __future__ import print_function
import os
from process import *
from final_python import god_function
import cv2

folder = './2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2/'
images_axial = [] # orientation (-z, x, y)
for img in os.listdir(folder):
    images.append(import_dicom(folder + img))

# NOTE: Refer to https://en.wikipedia.org/wiki/Anatomical_plane for orientation
temp = list(map(lambda img: cv2.resize(img, (176, 176)), images_axial))

# orientation (-x, -y, -z)
images_sagittal = map(lambda img: cv2.resize(img, (512, 512)),
    np.swapaxes(np.swapaxes(temp, 0, 1), 1, 2)[::-1, ::-1, :])

# orientation (-y, x, -z)
images_coronal = map(lambda img: cv2.resize(img, (512, 512)),
                     np.swapaxes(temp, 0, 2)[::-1])

print(god_function(images_axial, images_coronal, images_sagittal))

#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
from final_python import god_function
from process import import_dicom
import cv2

# NOTE: Refer to https://en.wikipedia.org/wiki/Anatomical_plane for orientation
dir_axial = './2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2/'
images_axial = [] # orientation (-z, x, y)
for img in sorted(os.listdir(dir_axial),
                  key=lambda img_name: int(img_name[2:]))[3:-2]:
    for _ in range(3):
        images_axial.append(import_dicom(dir_axial + img))
images_axial = np.array(images_axial[:-1])

# orientation (-x, -y, -z)
images_sagittal = np.swapaxes(np.swapaxes(images_axial, 0, 1),
                              1, 2)[::-1, ::-1, :]

# orientation (-y, x, -z)
images_coronal = np.swapaxes(images_axial, 0, 2)[::-1]

#dir_coronal = './2016.06.27 PVC Skull Model/Spiral Scan/DICOM/PA1/ST1/SE6/'
#images_coronal = []
#for img in sorted(os.listdir(dir_coronal),
                  #key=lambda img_name: int(img_name[2:]))[2:-1]:
    #for _ in range(3):
        #images_coronal.append(import_dicom(dir_coronal + img))
#images_coronal = np.array(images_coronal[:-1])

if __name__ == '__main__':
    print(god_function(images_axial, images_coronal, images_sagittal))

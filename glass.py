#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from final_python import god_function
from process import crop
import cv2

# NOTE: Refer to https://en.wikipedia.org/wiki/Anatomical_plane for orientation
# orientation (z, y, x); origin is corner of cube near right chin
dir_axial = './Fiducial data/Glass scan 1mm/Glass scan axial 1.25mm/'\
            'Patient-GLASS SCAN  1MM/Study_33455_CT_PHANTOM[20160526]/'\
            'Series_002_Plain Scan/'
images_axial = []
print("Importing DICOM...")
for img in sorted(os.listdir(dir_axial),
                  key=lambda img_name: int(img_name[2:-4]))[22:]:
    for _ in range(3):
        images_axial.append(crop(cv2.imread(dir_axial + img, 0)))

print("Forming arrays...")
pixel_spacing = list(map(float, images_axial[0].PixelSpacing))
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
    print("Starting detection...")
    xs, ys, zs = zip(*god_function(images_axial, images_coronal,
                                   images_sagittal))
    xs = np.int32(np.array(xs) * pixel_spacing[1])
    ys = np.int32(np.array(ys) * pixel_spacing[0])
    zs = np.int32(np.array(zs) * 512 * (pixel_spacing[0] / len(images_axial)))
    print(list(zip(xs, ys, zs)))


#!/usr/bin/env python
import dicom
import os
import numpy as np
from scipy.ndimage import gaussian_filter

folder = '/home/rharish/Programs/Python/Fiducial/2012.09.15 ACRELIC2 CT Scan '\
         'Data from ACTREC/09171420/'
images_axial = []
for dicom_img in sorted(os.listdir(folder), key=lambda img: int(img)):
    dic = dicom.read_file(folder + dicom_img)
    if 'AXIAL' in dic.ImageType:
        images_axial.append(dic)
        images_axial.append(dic)
images_axial = np.array([np.uint8(np.where(dic.pixel_array > 255, 255,
    dic.pixel_array)) for dic in sorted(images_axial,
    key=lambda dic: int(dic.SliceLocation))])

images_sagittal = np.array(list(map(lambda img: gaussian_filter(img, 1),
    np.swapaxes(np.swapaxes(images_axial, 0, 2), 1, 2)[:, ::-1, :])))

images_coronal = np.array(list(map(lambda img: gaussian_filter(img, 1),
    np.swapaxes(images_axial, 0, 1)[:, ::-1, :])))

images_axial = np.array(list(map(lambda img: gaussian_filter(img, 1),
                                 images_axial)))


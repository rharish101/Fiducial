#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from process import img_hist, func_minima, clahe_img, display, thresh_hist
import sys

def hough(images_axial, images_coronal, images_sagittal):
    centers = []
    for z, img in enumerate(images_axial):
        img = cv2.medianBlur(img, 5)
        img = thresh_hist(img)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            centers.extend([(i[1], i[0], z) for i in circles[0, :]])
        sys.stdout.write("\r" + str(z) + " images done out of " +\
                         str(len(images_axial)) + "\r")
        sys.stdout.flush()
    for y, img in enumerate(images_coronal):
        img = cv2.medianBlur(img, 5)
        img = thresh_hist(img)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            centers.extend([(i[1], y, len(images_axial) - i[0]) for i in\
                            circles[0, :]])
        sys.stdout.write("\r" + str(y) + " images done out of " +\
                         str(len(images_coronal)) + "\r")
        sys.stdout.flush()
    for x, img in enumerate(images_sagittal):
        img = cv2.medianBlur(img, 5)
        img = thresh_hist(img)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            centers.extend([(x, i[1], len(images_axial) - i[0]) for i in\
                            circles[0, :]])
        sys.stdout.write("\r" + str(z) + " images done out of " +\
                         str(len(images_sagittal)) + "\r")
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    return centers


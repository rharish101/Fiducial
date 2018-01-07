#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from process import img_hist, func_minima, clahe_img, display, thresh_hist
from refinement import *
import sys

def hough(images_axial, images_coronal, images_sagittal, verbose=False):
    centers_axial = []
    for z, img in enumerate(images_axial):
        img = cv2.medianBlur(img, 5)
        img = gaussian_filter(thresh_hist(img), 2)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None and len(circles[0]) <= 5:
            if verbose:
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                    #draw the center of the circle
                    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)
                    display(cimg, pause=0.5)

            circles = np.uint16(np.around(circles))
            centers_axial.extend([(i[1], i[0], z) for i in circles[0, :]])
        sys.stdout.write("\r" + str(z) + " images done out of " +\
                         str(len(images_axial)) + "\r")
        sys.stdout.flush()
    centers_coronal = []
    for y, img in enumerate(images_coronal):
        img = cv2.medianBlur(img, 5)
        img = thresh_hist(img)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None and len(circles[0]) <= 5:
            if verbose:
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                    #draw the center of the circle
                    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)
                    display(cimg, pause=0.5)

            circles = np.uint16(np.around(circles))
            centers_coronal.extend([(i[1], y, len(images_axial) - i[0]) for i in\
                            circles[0, :]])
        sys.stdout.write("\r" + str(y) + " images done out of " +\
                         str(len(images_coronal)) + "\r")
        sys.stdout.flush()
    centers_sagittal = []
    for x, img in enumerate(images_sagittal):
        img = cv2.medianBlur(img, 5)
        img = thresh_hist(img)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30, param1=50,
                                   param2=40, minRadius=10, maxRadius=0)

        if circles is not None and len(circles[0]) <= 5:
            if verbose:
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                    #draw the center of the circle
                    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),2)
                    display(cimg, pause=0.5)

            circles = np.uint16(np.around(circles))
            centers_sagittal.extend([(x, i[1], len(images_axial) - i[0]) for i in\
                            circles[0, :]])
        sys.stdout.write("\r" + str(x) + " images done out of " +\
                         str(len(images_sagittal)) + "\r")
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()
    if verbose:
        display.blank = True
    return refinement_axial(centers_axial, images_axial.shape[::-1]) +\
           refinement_coronal(centers_coronal, images_axial.shape[::-1]) +\
           refinement_saggital(centers_sagittal, images_axial.shape[::-1])


#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt

img_source = './fiducial.png'

img = cv2.imread(img_source, 0)
plt.title('Image')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# Smoothed Image Histogram and it's first minima
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
x = np.arange(len(hist))
smooth_hist = np.reshape(gaussian_filter1d(hist, 10), (len(hist)))
minima = 0
for i in x:
    if i > 0 and i < x[-1] and smooth_hist[i] <= smooth_hist[i - 1] and\
    smooth_hist[i] <= smooth_hist[i + 1]:
        minima = i
        break
plt.title('Image Histogram')
plt.plot(x, smooth_hist)
plt.plot(minima, smooth_hist[minima], 'rx')
plt.show()
new_img = np.array(img)

# Thresholding of image from 1st minima of histogram
# NOTE: This only works if background is noticeably darker than the brain
for i, row in enumerate(new_img):
    for j, val in enumerate(row):
        if val < minima:
            new_img[i][j] = 0
        else:
            new_img[i][j] = 255
new_img = gaussian_filter(new_img, 1)
plt.title('Thresholded Image')
plt.imshow(new_img, cmap='gray', vmin=0, vmax=255)
plt.show()
exit()

# Ratio Curve
summation = smooth_hist * range(1, 1 + len(smooth_hist))
ratio = [np.sum(summation[:i]) / np.sum(summation[(i + 1):]) for i in\
         range(len(summation) - 1)]
x = x[:-1]
plt.title('Ratio Curve')
plt.plot(x, ratio)
plt.show()

# Second-derivative of Normalized Rate Curve
y_first = np.diff(ratio) / np.diff(x)
x_first = 0.5 * (x[:-1] + x[1:])
y_second = np.diff(y_first) / np.diff(x_first)
x_second = 0.5 * (x_first[:-1] + x_first[1:])
second_der = gaussian_filter1d(y_second, 2)
plt.title('Normalized Rate Curve')
plt.plot(x_second, second_der)
#plt.imshow(smoothed_thresh, cmap='gray', vmin=0, vmax=255)
plt.show()


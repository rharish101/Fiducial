#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Smoothened Image Histogram
def img_hist(image, filter_sigma=10):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = np.reshape(gaussian_filter1d(hist, filter_sigma), (len(hist)))
    return hist

# Returns "1st" local minima of a function
def func_minima(func):
    for i in range(len(func)):
        if i > 0 and i < (len(func) -1) and func[i] <= func[i - 1] and\
        func[i] <= func[i + 1]:
            return i

# Second-derivative of Ratio Curve of Image Histogram: Normalized Rate Curve
def norm_rate_curve(hist, filter_sigma=2, verbose=False):
    summation = hist * range(1, 1 + len(hist))
    ratio = [np.sum(summation[:i]) / np.sum(summation[(i + 1):]) for i in\
            range(len(summation) - 1)]
    x = np.arange(len(ratio))
    if verbose:
        plt.title('Ratio Curve')
        plt.plot(x, ratio)
        plt.show()

    y_first = np.diff(ratio) / np.diff(x)
    x_first = 0.5 * (x[:-1] + x[1:])
    y_second = np.diff(y_first) / np.diff(x_first)
    second_der = gaussian_filter1d(y_second, filter_sigma)
    if verbose:
        x_second = 0.5 * (x_first[:-1] + x_first[1:])
        plt.title('Normalized Rate Curve')
        plt.plot(x_second, second_der)
        plt.show()
    return second_der

# Thresholding of image from 1st minima of histogram
# NOTE: This only works if background is noticeably darker than the brain
def thresh_hist(image, hist_filter_sigma=10, thresh_filter_sigma=2,
                verbose=False):
    hist = img_hist(image, filter_sigma=hist_filter_sigma)
    threshold = func_minima(hist)
    new_img = np.where(image>=threshold, 255 * np.ones(image.shape),
                       np.zeros(image.shape))
    new_img = gaussian_filter(new_img, thresh_filter_sigma)
    thresh_img = ((new_img.astype(np.float32) / new_img.max()) * 255).astype(
                 np.uint8)

    if verbose:
        plt.title('Image Histogram')
        plt.plot(np.arange(len(hist)), hist)
        plt.plot(threshold, hist[threshold], 'rx')
        plt.show()

        plt.title('Thresholded Image')
        plt.imshow(thresh_img, cmap='gray', vmin=0, vmax=255)
        plt.show()

    return thresh_img

# Canny-Edge detection
def canny_edge(image, edge_filter_sigma=2, binarize=False, verbose=False,
               **kwargs):
    if binarize:
        image = thresh_hist(image, verbose=verbose, **kwargs)
    edges = cv2.Canny(image, 100, 200)
    edges = gaussian_filter(edges, edge_filter_sigma)
    edges = ((edges.astype(np.float32) / edges.max()) * 255).astype(np.uint8)

    if verbose:
        plt.title('Image Edges')
        plt.imshow(edges, cmap='gray', vmin=0, vmax=255)
        plt.show()

    return edges

img_source = './fiducial.png'

img = cv2.imread(img_source, 0)
plt.title('Image')
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

canny_edge(img, binarize=True, verbose=True)


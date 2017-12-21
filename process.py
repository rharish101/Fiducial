#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Display grayscale image
def display(image, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)

# Contrast Limited Adaptive Histogram Equalization
def clahe_img(image, clipLimit=2.0, tileGridSize=(8, 8), verbose=False):
    improved = cv2.createCLAHE(clipLimit=clipLimit,
                               tileGridSize=tileGridSize).apply(image)
    if verbose:
        display(improved, 'After CLAHE')
    return improved

# Smoothened Image Histogram
def img_hist(image, hist_filter_sigma=2):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = np.reshape(hist, (len(hist)))
    return gaussian_filter1d(hist, hist_filter_sigma)

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
def thresh_hist(image, thresh_filter_sigma=2.7, clahe=True, verbose=False,
                **kwargs):
    if clahe:
        image = clahe_img(image, verbose=verbose)
    hist = img_hist(image, **kwargs)
    threshold = func_minima(hist)
    new_img = np.where(image>=threshold, 255 * np.ones(image.shape),
                       np.zeros(image.shape))
    new_img = gaussian_filter(new_img, thresh_filter_sigma)
    thresh_img = ((new_img.astype(np.float32) / new_img.max()) * 255).astype(
                 np.uint8)

    if verbose:
        plt.figure()
        plt.title('Image Histogram')
        plt.plot(np.arange(len(hist)), hist)
        plt.plot(threshold, hist[threshold], 'rx')
        plt.show(block=False)

        display(thresh_img, 'Thresholded Image')

    return thresh_img

# Adaptive gaussian thresholding
def thresh_adaptive(image, binarize='mean', blocksize=17, thresh_C=6.5,
                    thresh_filter_sigma=1, clahe=True, verbose=False):
    if clahe:
        image = clahe_img(image, verbose=verbose)

    if binarize == 'mean':
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif binarize == 'gaussian':
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError('Invalid option for binarization')

    raw_thresh = cv2.adaptiveThreshold(image, 255, method,
                                       cv2.THRESH_BINARY, blocksize, thresh_C)
    smooth_thresh = gaussian_filter(raw_thresh, thresh_filter_sigma)
    thresh_img = ((smooth_thresh.astype(np.float32) / smooth_thresh.max()) *\
                  255).astype(np.uint8)

    if verbose:
        display(thresh_img, 'Thresholded Image')

    return thresh_img

# Thresholding with Otsu's Binarization
def thresh_otsu(image, thresh_filter_sigma=2, clahe=True, verbose=False):
    if clahe:
        image = clahe_img(image, verbose=verbose)
    _, raw_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +\
                                  cv2.THRESH_OTSU)
    smooth_thresh = gaussian_filter(raw_thresh, thresh_filter_sigma)
    thresh_img = ((smooth_thresh.astype(np.float32) / smooth_thresh.max()) *\
                  255).astype(np.uint8)

    if verbose:
        display(thresh_img, 'Thresholded Image')

    return thresh_img

# Canny-Edge detection
def canny_edge(image, edge_filter_sigma=4, binarize='histogram', verbose=False,
               **kwargs):
    if binarize == 'histogram':
        image = thresh_hist(image, verbose=verbose, **kwargs)
    elif binarize in ('gaussian', 'mean'):
        image = thresh_adaptive(image, binarize=binarize, verbose=verbose,
                                **kwargs)
    elif binarize == 'otsu':
        image = thresh_otsu(image, **kwargs)
    elif binarize is not None and not (type(binarize) == str and\
    binarize.lower() == 'none'):
        raise ValueError('Invalid option for binarization')

    edges = cv2.Canny(image, 100, 200)
    edges = gaussian_filter(edges, edge_filter_sigma)
    edges = ((edges.astype(np.float32) / edges.max()) * 255).astype(np.uint8)

    if verbose:
        display(edges, 'Image Edges')

    return edges

# Longest-edge from contours in image
def longest_edge(image, canny=True, outline_filter_sigma=2, verbose=False,
                 **kwargs):
    if canny:
        image = canny_edge(image, verbose=verbose, **kwargs)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    max_len = 0
    outline = None
    for contour in contours:
        if contour.shape[0] > max_len:
            max_len = contour.shape[0]
            outline = contour
    outline_img = cv2.drawContours(np.zeros(image.shape), [outline], -1,
                                   (255, 255, 255), 2)
    smooth_outline = gaussian_filter(outline_img, outline_filter_sigma)
    outline_img = ((smooth_outline.astype(np.float32) / smooth_outline.max(
                    )) * 255).astype(np.uint8)

    if verbose:
        display(outline_img, 'Longest Edge')
        
    return outline_img

# Harris Corners
def harris_corners(image, outline=True, blockSize=2, ksize=3, harris_k=0.06,
                   verbose=False, **kwargs):
    if outline:
        image = longest_edge(image, verbose=verbose, **kwargs)

    raw_corners = cv2.cornerHarris(image, blockSize, ksize, harris_k)
    dilated = cv2.dilate(raw_corners, None)
    _, scaled = cv2.threshold(dilated, 0.01 * dilated.max(), 255, 0)
    corners = scaled.astype(np.uint8)

    if verbose:
        display(corners, 'Outline corners')

    return corners

img_source = './fiducial.png'

img = cv2.imread(img_source, 0)
display(img, 'Image')

if __name__ == '__main__':
    harris_corners(img, verbose=True)

plt.show()


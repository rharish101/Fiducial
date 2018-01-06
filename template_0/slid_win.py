#!/usr/bin/env python
from __future__ import print_function
from scipy.misc import imread, imresize
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt

# Display grayscale image
def display(image, title=None, pause=None):
    if pause is None:
        plt.figure()
    if title is not None:
        plt.title(title)

    if display.blank or pause is None:
        display.image = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        display.blank = False
    else:
        display.image.set_data(image)

    if pause:
        plt.pause(pause)
    else:
        plt.show(block=False)
display.blank = True
display.image = None

# Generator to yield sliding windows
def windows(image, window_size, stride_size):
    if np.any(np.array(image.shape) < np.array(window_size)):
        return

    for row in range(0, image.shape[0] - window_size[0] + 1, stride_size[0]):
        for col in range(0, image.shape[1] - window_size[1] + 1,
                         stride_size[1]):
            yield (row, row + window_size[0], col, col + window_size[1])

# Image Pyramids to rescale images
def pyramid(image, resize_factor, min_size):
    return [image]
    #while np.any(np.array(image.shape) >= np.array(min_size)):
        #yield image
        #image = imresize(image, int((1.0 / resize_factor) * 100))

# Non maximum suppression to remove overlapping sliding windows
def non_max_suppr(box1_coord, box2_coord, overlap_threshold_area):
    if box1_coord[1] <= box2_coord[0] or box2_coord[1] <= box1_coord[0] or\
    box1_coord[3] <= box2_coord[2] or box2_coord[3] <= box1_coord[0]:
        return False
    else:
        x_list = [box1_coord[0], box1_coord[1], box2_coord[0], box2_coord[1]]
        x_list.sort()
        width = x_list[2] - x_list[1]
        y_list = [box1_coord[2], box1_coord[3], box2_coord[2], box2_coord[3]]
        y_list.sort()
        height = y_list[2] - y_list[1]
        area = width * height
        if area > overlap_threshold_area:
            return True
        else:
            return False

# Contrast Limited Adaptive Histogram Equalization
def clahe_img(image, clipLimit=2.0, tileGridSize=(8, 8)):
    improved = cv2.createCLAHE(clipLimit=clipLimit,
                               tileGridSize=tileGridSize).apply(image)
    return improved.astype(np.uint8)

# Uses Sliding Windows and yields images
def sliding_windows(image, model_name, model_input_shape, label,
                    window_size=(30, 30), stride_size=(10, 10),
                    resize_factor=1.25, min_size=(100, 100), clahe=False,
                    overlap_threshold=0.9, border_colour=None,
                    return_images=False, verbose=False):
    model = load_model(model_name)
    images = []

    for i, mini_image in enumerate(pyramid(image, resize_factor, min_size)):
        for micro_image_box in windows(mini_image, window_size, stride_size):
            micro_image = mini_image[micro_image_box[0]:micro_image_box[1],
                                     micro_image_box[2]:micro_image_box[3]]

            if verbose:
                display(micro_image, pause=0.01)

            if clahe:
                micro_image = clahe_img(micro_image)

            if border_colour is not None:
                border_x = int(micro_image.shape[0] * 0.05)
                border_y = int(micro_image.shape[1] * 0.05)
                micro_image = imresize(micro_image, (micro_image.shape[0] -\
                    2 * border_x, micro_image.shape[1] - 2 * border_y))
                micro_image = cv2.copyMakeBorder(micro_image, border_x,
                                                 border_x, border_y, border_y,
                                                 cv2.BORDER_CONSTANT,
                                                 value=border_colour)

            prediction = model.predict(np.resize(gaussian_filter(micro_image,
                1), model_input_shape), batch_size=1)
            if int(prediction[0] >= 0.5) == label:
                images.append(np.array((int(pow(resize_factor, i)) *\
                                        micro_image_box, micro_image)))
        if verbose:
            print("Pyramid level " + str(i + 1) + " done")

    if verbose:
        print("Starting trimming...")
    to_delete = []
    for i in range(len(images))[::-1]:
        for j in range(len(images))[::-1]:
            if j >= i:
                continue
            if non_max_suppr(images[i][0], images[j][0], overlap_threshold_area=
            overlap_threshold * window_size[0] * window_size[1]):
                to_delete.append(j)
    images = [images[i] for i in range(len(images)) if i not in\
              to_delete]
    if verbose:
        print("Trimming done")

    if return_images:
        return [micro_image[1] for micro_image in images]
    else:
        return np.array([(sum(micro_image[0][:2]) / 2,
            sum(micro_image[0][2:]) / 2) for micro_image in images])


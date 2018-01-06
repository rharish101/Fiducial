from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt

def img_hist(image, hist_filter_sigma=2):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = np.reshape(hist, (len(hist)))
    return gaussian_filter1d(hist, hist_filter_sigma)

def func_minima(func):
    for i in range(len(func)):
        if i > 0 and i < (len(func) -1) and func[i] <= func[i - 1] and\
        func[i] <= func[i + 1]:
            return i

def clahe_img(image, clipLimit=2.0, tileGridSize=(8, 8), verbose=False):
    improved = cv2.createCLAHE(clipLimit=clipLimit,
                               tileGridSize=tileGridSize).apply(image)
    if verbose:
        display(improved, 'After CLAHE')
    return improved.astype(np.uint8)

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


def thresh_hist(image, thresh_filter_sigma=2.7, clahe=True,verbose=False,
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

img = cv2.imread('./Desktop/10.png',0)
img = cv2.medianBlur(img,5)
img=thresh_hist(img)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#hyperparameters for acrelic 1
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,30,
                            param1=50,param2=40,minRadius=10,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

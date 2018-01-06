#!/usr/bin/env python
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from process import img_hist, func_minima, clahe_img, display, thresh_hist

img = cv2.imread('./acrelic1/tp_21.png',0)
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

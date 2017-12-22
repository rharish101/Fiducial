#!/usr/bin/env python
import os
from process import *

folder = "./Fiducial data/Glass scan 1mm/Glass scan axial 1.25mm/"\
         "Patient-GLASS SCAN  1MM/Study_33455_CT_PHANTOM[20160526]/"\
         "Series_002_Plain Scan/"

for img_name in os.listdir(folder):
    img = cv2.imread(folder + img_name, 0)[67:450, :]
    display(img, img_name)
    corners = harris_corners(img, outline=False)
    display(corners, 'Corners')
    try:
        plt.show()
    except KeyboardInterrupt:
        break


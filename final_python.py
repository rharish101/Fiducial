#!/usr/bin/env python
import numpy as np
from template_0.slid_win import sliding_windows
from process import shi_tomasi

def template2(img):
    return shi_tomasi(img, maxCorners=10, qualityLevel=0.25)

def god_function(list_axial, list_coronal, list_sagittal): 
    length = len(list_axial)

    corners = []
    for z in range(63, length):
        if z in range(90, 111):
            continue
        corners.extend(template2(list_axial[z]))

    return refinement_axial(corners, list_axial.shape[::-1], mode='soft')


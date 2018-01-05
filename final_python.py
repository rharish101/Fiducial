#!/usr/bin/env python
import numpy as np
from template_0.slid_win import sliding_windows
from process import shi_tomasi

def template1(img):
    return sliding_windows(img, './template_0/template_0_detect.h5', (50, 50),
                           1)

def template2(img):
    #x,y are the coordinates of the center of the valley
    #returns true if template2 is found and sets Ma[x][y][index]=2 if plane='axial'...    
    return shi_tomasi(img)

def god_function(list_axial, list_coronal, list_sagittal): 
    length = len(list_axial)
    Ma = np.zeros(list_axial.shape, dtype=np.uint8)
    Mc = np.zeros(list_axial.shape, dtype=np.uint8)
    Ms = np.zeros(list_axial.shape, dtype=np.uint8)

    fidu_coordinates = []
    for z in range(length):
        for y, x in template1(list_axial[z]):
            #if (length - z), y in template2(list_sagittal[x]) and\
            #(length - z), x in template2(list_coronal[y]):
                #fidu_coordinates.append((x, y, z))
            Ma[x, y, index] = 1
        for y, x in template2(list_axial[index]):
            #if ((length - z), y in template1(list_sagittal[x]) or\
            #(length - z), y in template2(list_sagittal[x])) and\
            #((length - z), x in template1(list_coronal[y]) or\
            #(length - z), x in template2(list_coronal[y])):
                #fidu_coordinates.append((x, y, z))
            Ma[x, y, index] = 2

    for a in range(length):
        for b in range(length):
            for c in range(length):
                if Ma[a][b][c] == 1 and\
                len(template2(list_sagittal[a])) > 0 and\
                len(template2(list_coronal[b])) > 0:
                    fidu_coordinates.append((a, b, c))
                elif Ma[a][b][c] == 2 and\
                (len(template1(list_sagittal[a])) > 0 or\
                len(template2(list_sagittal[a])) > 0) and\
                (len(template1(list_coronal[a])) > 0 or\
                len(template2(list_coronal[a])) > 0):
                    fidu_coordinates.append((a, b, c))

    return fidu_coordinates


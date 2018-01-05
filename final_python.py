#!/usr/bin/env python
import numpy as np
from template_0.slid_win import sliding_windows
from process import shi_tomasi

def template1(img):
    return sliding_windows(img, './template_0/template_0_detect.h5', (50, 50),
                           1)

def template2(img):
    return shi_tomasi(img)

def god_function(list_axial, list_coronal, list_sagittal): 
    length = len(list_axial)
    Ma = np.zeros(list_axial.shape, dtype=np.uint8)

    fidu_coordinates = []
    for z in range(length):
        for y, x in template1(list_axial[z]):
            #if (length - z), y in template2(list_sagittal[x]) and\
            #(length - z), x in template2(list_coronal[y]):
                #fidu_coordinates.append((x, y, z))
            Ma[x, y, z] = 1
        for y, x in template2(list_axial[z]):
            #if ((length - z), y in template1(list_sagittal[x]) or\
            #(length - z), y in template2(list_sagittal[x])) and\
            #((length - z), x in template1(list_coronal[y]) or\
            #(length - z), x in template2(list_coronal[y])):
                #fidu_coordinates.append((x, y, z))
            Ma[x, y, z] = 2
            
    for x in range(length):
        for x in range(length):
            for z in range(length):
                #if there is series of 1's then this code makes makes the 1's after the first 1 to be 0
                    if Ma[x,y,z]==1:
                    z+=1
                    while(Ma[x,y,z]==1):
                        Ma[x,y,z]=0
                        z+=1

    for x in range(length):
        for y in range(length):
            for z in range(length):
                if Ma[x][y][z] == 1 and\
                ((length - z), y) in template2(list_sagittal[x]) and\
                ((length - z), x) in template2(list_coronal[y]):
                    fidu_coordinates.append((x, y, z))
                elif Ma[x][y][z] == 2 and\
                (((length - z), y) in template1(list_sagittal[x]) or\
                ((length - z), y) in template2(list_sagittal[x])) and\
                (((length - z), x) in template1(list_coronal[y]) or\
                ((length - z), x) in template2(list_coronal[y])):
                    fidu_coordinates.append((x, y, z))

    return fidu_coordinates


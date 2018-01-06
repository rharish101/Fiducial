#!/usr/bin/env python
import numpy as np
from template_0.slid_win import sliding_windows
from process import shi_tomasi

def template1(img):
    return sliding_windows(img, './template_0/template_0_detect.h5',
                           (1, 50, 50, 1), 1)

def template2(img):
    #return shi_tomasi(img)
    return sliding_windows(img, './template_0/template_0_detect.h5',
                           (1, 40, 40, 1), 1)

def refine(list_):
    refined_list=[list_[0]]
    i=1
    while(i)<len(list_):
         if list_[i]!=list_[i-1]:
            refined_list.append(list_[i])
         i+=1
    return refined_list


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
        for y in range(length):
            z = 0
            while z < length:
                if Ma[x, y, z] == 1:
                    z += 1
                    while(Ma[x, y, z] == 1):
                        Ma[x,y,z] = 0
                        z += 1
                z += 1
    for z in range(length):
        #list_valley(), list_single are the lists of coordinates of the valley and single peaks in list_axial(z)
       for y,x in list_valley:
           for z1 in range(length):
             l=[]
             if (y,x) in list_valley(list_axial[z1]):
                 l.append(2)
             elif (y,x) in list_single(list_axial[z1]):
                 l.append(1)
             else
                 l.append(0)
           final_str=refine(l)
           if final_str==[0,1,2,1,0] or final_str==[0,1,2,1] or final_str==[1,2,1,0] or final_str==[1,2,1]:
              Ma[x,y,z]=2
            
                

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


#!/usr/bin/env python
from __future__ import print_function
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

def god_function(list_axial,list_coronal,list_saggital): 
    x_dim=176
    y_dim=176
    z_dim=176

    Ma=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
    Mc=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
    Ms=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)

    fidu_coordinates=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
    for index in range(1, 1 + len(list_axial)):
        template1(list_axial[index - 1])
        template2(list_axial[index - 1])

    for a in range(x_dim):
        for b in range(y_dim):
            for c in range(z_dim):
                if Ms[a][b][c]==1:
                    if template2(list_saggital[a])['Bool'] is True and template2(list_coronal[b])['Bool'] is True:
                        print('Fiducial present at (a,b,c)')
                        fidu_coordinates[a][b][c]=1

    for a1 in range(x_dim):
        for b1 in range(y_dim):
            for c1 in range(z_dim):
                if Ms[a1][b1][c1]==2:
                    if (template1(list_saggital[a1])['Bool'] or template2(list_saggital[a1])['Bool']) and (template1(list_saggital[b1])['Bool'] or template2(list_coronal[b1])['Bool']):
                        print('Fiducial present at (a1,b1,c1)')
                        fidu_coordinates[a1][b1][c1]=1

    return  fidu_coordinates      





        


  

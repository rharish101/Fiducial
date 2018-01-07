#!/usr/bin/env python
def refinement_axial(l, shape, mode='extreme'):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x//7,y//7,z))
    for i in range(x_dim//7):
        for j in range(y_dim//7):
            k = 0
            while k < z_dim:
                if (i,j,k) in new_l:
                    k+=1
                    if mode == 'extreme':
                        while k < z_dim:
                            if (i,j,k) in new_l:
                                new_l.remove((i,j,k))
                            k+=1
                    else:
                        while (i,j,k) in new_l:
                            new_l.remove((i,j,k))
                            k+=1
                k+=1
                      
    final_l=[]
    for (x,y,z) in new_l:
         final_l.append((7*x,7*y,z))
    return final_l

def refinement_coronal(l, shape):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x//7,y,z//7))
    for i in range(x_dim//7):
        for k in range(z_dim//7):
            j = 0
            while j < y_dim:
                if (i,j,k) in new_l:
                    j+=1 
                    while j < y_dim:
                        if (i,j,k) in new_l: 
                            new_l.remove((i,j,k))
                        j+=1
                j+=1
    final_l=[]
    for (x,y,z) in new_l:
          final_l.append((7*x,y,7*z))
    return final_l

def refinement_saggital(l, shape):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x,y//7,z//7))
    for j in range(y_dim//7):
         for k in range(z_dim//7):
             i = 0
             while i < x_dim:
                 if (i,j,k) in new_l:
                    i+=1 
                    while i < x_dim:
                        if (i,j,k) in new_l: 
                            new_l.remove((i,j,k))
                        i+=1
                 i+=1
    final_l=[]
    for (x,y,z) in new_l:
          final_l.append((x,7*y,7*z))
    return final_l

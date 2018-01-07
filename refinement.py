#!/usr/bin/env python
def refinement_axial(l, shape):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x//5,y//5,z))
    for i in range(x_dim//5):
        for j in range(y_dim//5):
            k = 0
            while k < z_dim:
                if (i,j,k) in new_l:
                    k+=1
                    while (i,j,k) in new_l:
                        new_l.remove((i,j,k))
                        k+=1
                k+=1
                      
    final_l=[]
    for (x,y,z) in new_l:
         final_l.append((5*x,5*y,z))
    return final_l

def refinement_coronal(l, shape):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x//5,y,z//5))
    for i in range(x_dim//5):
        for k in range(z_dim//5):
            j = 0
            while j < y_dim:
                if (i,j,k) in new_l:
                    j+=1 
                    while (i,j,k) in new_l: 
                        new_l.remove((i,j,k))
                        j+=1
                j+=1
    final_l=[]
    for (x,y,z) in new_l:
          final_l.append((5*x,y,5*z))
    return final_l

def refinement_saggital(l, shape):
    new_l=[]
    x_dim, y_dim, z_dim = shape
    for (x,y,z) in l:
        new_l.append((x,y//5,z//5))
    for j in range(y_dim//5):
         for k in range(z_dim//5):
             i = 0
             while i < x_dim:
                 if (i,j,k) in new_l:
                    i+=1 
                    while (i,j,k) in new_l: 
                         new_l.remove((i,j,k))
                         i+=1
                 i+=1
    final_l=[]
    for (x,y,z) in new_l:
          final_l.append((x,5*y,5*z))
    return final_l

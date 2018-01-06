def refinement_axial(l):
     new_l=[]
     for (x,y,z) in l:
         new_l.append((x/5,y/5,z/5))
     for i in range(x_dim/5):
         for j in range(y_dim/5):
             for k in range(z_dim/5):
                 if (i,j,k) in new_l:
                    k+=1
                    while (i,j,k) in new_l:
                         k+=1    
                         new_l.remove((i,j,k))
                      
     final_l=[]
     for (x,y,z) in new_l:
          final_l.append(5*x,5*y,5*z)                    
     return final_l

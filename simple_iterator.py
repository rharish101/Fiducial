import numpy as np

def template1(img)
#   returns the list of coordinates where template1 is discovered 
return list_coordinates

#iterate in three directions and mark 1 if template 1 is found
def new_god(list_axial,list_saggital,list_coronal):
    length=len(list_axial)
    m=np.zeros((length,length,length),dtype=np.int8)  
    fidu_coordinates = []
        
    for x in range(length):
      for y in range(length):
          for z in range(length):
              if (y,x) in template1(list_axial[z]) or (length-z,y) in template1(list_saggital[x]) or (length-z,x) in template1(list_coronal[y]):
                 m[x,y,z]=1
                  
     #refining over axial images
     for i in range(length):
        for j in range(length):
            k = 0
            while k < length:
                if m[i,j,k] == 1:
                    fidu_coordinates.append(i,j,k)
                    k += 1
                    while(m[i,j,k] == 1):
                        m[i,j,k] = 0
                        k += 1
     #refining over saggital images           
     for i in range(length):
        for j in range(length):
            k=0
            if m[k,i,j]==1:
                    fidu_coordinates.append(k,i,j) 
                    k+=1
                    while(m[k,i,j] == 1):
                        m[k,i,j] = 0
                        k += 1 
     #refining over coronal images
     for i in range(length):
        for j in range(length):
            k=0
            if m[j,k,i]==1:
               fidu_coordinates.append(j,k,i)    
               k+=1
               while(m[j,k,i] == 1):
                        m[j,k,i] = 0
                        k += 1 
                  
     return fidu_coordinates

           
                 


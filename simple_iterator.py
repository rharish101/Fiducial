import numpy as np

def template1(img)
#   returns the list of coordinates where template1 is discovered 
return list_coordinates

#iterate in three directions and mark 1 if template 1 is found
def new_god(list_axial,list_saggital,list_coronal):
    length=len(list_axial)  
    fidu_coordinates = []
        
    for x in range(length):
      for y in range(length):
          for z in range(length):
              if (y,x) in template1(list_axial[z]) or (length-z,y) in template1(list_saggital[z]) or (length-z,x) in template1(list_coronal[y]):
                 fidu_coordintes.append(x,y,z)      
      


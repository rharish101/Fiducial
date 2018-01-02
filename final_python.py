#python 2.7
import numpy as np

#enter the dimensions of the image
x_dim=
y_dim=
z_dim=

Ma=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
Mc=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
Ms=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)

fidu_coordinates=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)


#given plane and index of the slice this function outputs the image 
def image(plane,index):

return image


def template1(plane,index):
#Boolean is true if image has template1 
#x,y are the coordinates of the center of concentric circles and sets Ma[x][y][index]=1 is plane='axial'....
return Boolean,x,y

def template2(plane,index):
#x,y are the coordinates of the center of the valley
#returns true if template2 is found and sets Ma[x][y][index]=2 if plane='axial'...    
return Boolean,x,y

#main code starts:
index=1
template1('axial',1)=False
template2('axial',1)=False
#N is the no. of slices of each view
while index is not equal to N and template1('axial',index) is not True and template2('axial',index) is not True:
 template1('axial',index)
 template2('axial',index)
 index++
 
#some issues reagarding last if statement
for a in range(x_dim):
    for b in range(y_dim):
        for c in range(z_dim):
            if Ms[a][b][c]==1:
               if template2('saggital',a) is True and template2('coronal',b) is True:
                  print 'Fiducial present at (x,y,z)'
                  fidu_coordinates[x][y][z]=1  

for a1 in range(x_dim):
    for b1 in range(y_dim):     
        for c1 in range(z_dim): 
            if Ms[a1][b1][c1]==2:
               if (template1('saggital',a1) or template2('saggital',a1)) and (template1('saggital',b1) or template2('coronal',b1) is True:
                  print 'Fiducial present at (x,y,z)'
                  fidu_coordinates[x][y][z]=1
                                                                              

           





        


  

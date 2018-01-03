#python 2.7
import numpy as np

#enter the dimensions of the image
x_dim=
y_dim=
z_dim=

Ma=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
Mc=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
Ms=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)




#given plane and index of the slice this function outputs the image 
def image(plane,index):
    if plane=='axial':
       image=list_axial[index-1]
    elif plane=='coronal':
       image=list_coronal[index-1]
    else:
       image=list_saggital[index-1]

return image


def template1(plane,index):
   image=image(plane,index)
#Boolean is true if image has template1 
#x,y are the coordinates of the center of concentric circles and sets Ma[x][y][index]=1 is plane='axial'....
return {'Bool':,'x_coor':,'y_coor':}

def template2(plane,index):
   image=image(plane,index)
#x,y are the coordinates of the center of the valley
#returns true if template2 is found and sets Ma[x][y][index]=2 if plane='axial'...    
return {'Bool': ,'x_coor': ,'y_coor': }

def god_function(list_axial,list_coronal,list_saggital): 
        
        fidu_coordinates=np.zeros((x_dim,y_dim,z_dim),dtype=np.float32)
        #N is the no of slices
        index=1
        while index is not  N:
         template1('axial',index)
         template2('axial',index)
         index=index+1


        for a in range(x_dim):
            for b in range(y_dim):
                for c in range(z_dim):
                    if Ms[a][b][c]==1:
                       if template2('saggital',a)['Bool'] is True and template2('coronal',b)['Bool'] is True:
                          print 'Fiducial present at (a,b,c)'
                          fidu_coordinates[a][b][c]=1

        for a1 in range(x_dim):
            for b1 in range(y_dim):
                for c1 in range(z_dim):
                    if Ms[a1][b1][c1]==2:
                       if (template1('saggital',a1)['Bool'] or template2('saggital',a1)['Bool']) and (template1('saggital',b1)['Bool'] or template2('coronal',b1)['Bool']:
                         print 'Fiducial present at (a1,b1,c1)'
                         fidu_coordinates[a1][b1][c1]=1

return  fidu_coordinates      





        


  

from PIL import Image
import numpy as np
height=15
width=15
volume=[[0.0 for i in range (width)] for j in range (height)]

tab_object=open("tab1.txt","r");
for i in range(14):
    for j in range(14):
        print tab_object.readline()
        volume[i][j]=tab_object.readline()

pic=np.array(volume).astype(np.uint8)
im2=Image.fromarray(pic2)
#im2.convert('RGB')
im2.save("image_after_conv.png")

import functions as f
import layers as l

#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)

from PIL import Image
import numpy as np
weights1=[[0.02,0.02,0.04,0.01],[0.02,0.03,0.03,0.02],[0.02,0.05,0.02,0.01],[0.01,0.01,0.03,0.05]]
weights2=[[0.1,0.2],[0.3,0.35]]
layer1=l.ConvLayer(452,4,2,0,3,1,weights1)
layer2=l.ConvLayer(225,2,5,0,1,1,weights2)
im=Image.open("cat_origin.jpeg")
pic = np.array(im)
height=len(pic)
width=len(pic[0])
volume=[[[0 for i in range (width)] for j in range (height)] for k in range (3)]
for i in range(height):
    for j in range(width):
        for k in range(3):
            volume[k][i][j]=pic[i][j][k]
volume2=layer1.forward(volume)
pic2=np.array(volume2[0]).astype(np.uint8)
im2=Image.fromarray(pic2)
#im2.convert('RGB')
im2.save("cat_int.png")
volume3=layer2.forward(volume2)
pic3=np.array(volume3[0]).astype(np.uint8)
im3=Image.fromarray(pic3)
im3.save("cat_final.png")

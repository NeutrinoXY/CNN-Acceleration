import functions as f
import layers as l

#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)
layer=l.ConvLayer(1,2,3,4,5,6)
print(layer.fieldSize)

from PIL import Image
import numpy as np
layer=l.ConvLayer(128,3,2,0,3,1)
im=Image.open("animal-chat-icone-4095-128.png")
pic = np.array(im)
height=len(pic)
width=len(pic[0])
volume=[[[0 for i in range width] for j in range height] for k in range 3]
for i in range(height):
    for j in range(width):
        for k in range(3):
            volume[k][i][j]=pic[i][j][k]
pic2=np.array(volume)
pic2.save("cat.png")

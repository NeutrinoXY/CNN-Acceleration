import functions as f
import layers as l
from weights import *

#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)

from PIL import Image
import numpy as np
#weights1=[[0.03,0.02,0.04,0.01],[0.02,0.03,0.03,0.02],[0.02,0.05,0.02,0.01],[0.01,0.01,0.03,0.05]]
weights2=[[0.1,0.2],[0.3,0.35]]

#layer1=l.ConvLayer(452,4,2,0,3,1,weights1)
layer1=l.ConvLayer(24,3,1,1,3,64,weights_conv_1,biases_conv_1)
layer2=l.ReluLayer(24,64)
layer3=l.MaxpoolLayer(24,3,2,1,64)
layer4=l.ConvLayer(12,3,1,1,64,32,weights_conv_2,biases_conv_2)
layer5=l.ReluLayer(12,32)
layer6=l.MaxpoolLayer(12,3,2,1,32)
layer7=l.ConvLayer(6,3,1,1,32,20,weights_conv_3,biases_conv_3)
layer8=l.ReluLayer(6,20)
layer9=l.MaxpoolLayer(6,3,2,1,20)
layer10=l.FullyConnected(3,10,20,weights_fc,biases_fc)
#im=Image.open("cat_origin.jpeg")
im=Image.open("automobile1.png")
pic = np.array(im)
height=len(pic)
width=len(pic[0])
volume=[[[0 for i in range (24)] for j in range (24)] for k in range (3)]
for i in range(24):
    for j in range(24):
        for k in range(3):
            volume[k][i][j]=pic[i+4][j+4][k]
average=[0,0,0]
for i in range(3):
    for j in range(24):
        for k in range(24):
            average[i]+=volume[i][j][k]
    average[i]=average[i]/(24*24)
print(average)
deviation=[0,0,0]
for i in range(3):
    for j in range(24):
        for k in range(24):
            deviation[i]+=(volume[i][j][k]-average[i])**2
    deviation[i]=(deviation[i]/(24*24))**0.5
print(deviation)
for i in range(3):
    for j in range(24):
        for k in range(24):
            print(volume[i][j][k])
            #volume[i][j][k]=(volume[i][j][k]-average[i])/max(deviation[i],(1/(24*24))**0.5)
            print(volume[i][j][k])
volume2=layer1.forward(volume)
print(volume2[0])
pic2_0=np.array(volume2[0]).astype(np.uint8)
pic2_1=np.array(volume2[1]).astype(np.uint8)
im2_0=Image.fromarray(pic2_0)
im2_1=Image.fromarray(pic2_1)
#im2.convert('RGB')
im2_0.save("airplane_int_0.png")
im2_1.save("airplane_int_1.png")
volume3=layer2.forward(volume2)
print(volume3[0])
volume4=layer3.forward(volume3)
volume5=layer4.forward(volume4)
pic3_0=np.array(volume5[0]).astype(np.uint8)
pic3_1=np.array(volume5[1]).astype(np.uint8)
im3_0=Image.fromarray(pic3_0)
im3_1=Image.fromarray(pic3_1)
im3_0.save("airplane_int2_0.png")
im3_1.save("airplane_int2_1.png")
volume6=layer5.forward(volume5)
volume7=layer6.forward(volume6)
volume8=layer7.forward(volume7)
volume9=layer8.forward(volume8)
volume10=layer9.forward(volume9)
volume11=layer10.forward(volume10)
print(volume11)
SoftMax=l.SoftMax()
volume12=SoftMax.forward(volume11)
print(volume12)

#pic3=np.array(volume3[0]).astype(np.uint8)
#im3=Image.fromarray(pic3)
#im3.save("cat_final_0.png")

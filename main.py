import layers as layers
from weights import *

#f.Conv2D([1,2,3,4,5,6,7],3,1,0,1,0)
#f.Conv2D([1,2,3,4,5,6,7],3,2,0,1,0)

from PIL import Image
import numpy as np

def Normalize(volume):
    average=[0,0,0]
    for i in range(3):
        for j in range(24):
            for k in range(24):
                average[i]+=volume[i][j][k]
        average[i]=average[i]/(24*24)
    #print(average)
    deviation=[0,0,0]
    for i in range(3):
        for j in range(24):
            for k in range(24):
                deviation[i]+=(volume[i][j][k]-average[i])**2
        deviation[i]=(deviation[i]/(24*24))**0.5
    #print(deviation)
    for i in range(3):
        for j in range(24):
            for k in range(24):
                #print(" ")
                #print(volume[i][j][k])
                volume[i][j][k]=(volume[i][j][k]-average[i])/max(deviation[i],(1/(24*24))**0.5)
                #print(volume[i][j][k])
                #volume[i][j][k]=volume[i][j][k]
                #print(volume[i][j][k])
    return volume

def RunCNN(volume,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8,layer9,layer10):
    pic=np.array(volume[0]).astype(np.uint8)
    im=Image.fromarray(pic)
    im.save("img_origine.png")
    volume2=layer1.forward(volume)
    pic2_0=np.array(volume2[0]).astype(np.uint8)
    im2_0=Image.fromarray(pic2_0)
    im2_0.save("img_int_0.png")
    volume3=layer2.forward(volume2)
    volume4=layer3.forward(volume3)
    #print(volume4)
    volume5=layer4.forward(volume4)
    pic3_0=np.array(volume5[0]).astype(np.uint8)
    im3_0=Image.fromarray(pic3_0)
    im3_0.save("img_int2_0.png")
    volume6=layer5.forward(volume5)
    volume7=layer6.forward(volume6)
    volume8=layer7.forward(volume7)
    volume9=layer8.forward(volume8)
    volume10=layer9.forward(volume9)
    volume11=layer10.forward(volume10)
    SoftMax=layers.SoftMax()
    volume12=SoftMax.forward(volume11)
    print(volume12)
    indiceMax=0
    valeurMax=0
    for i in range(len(volume11)):
        if(volume12[i]>valeurMax):
            valeurMax=volume12[i]
            indiceMax=i
    SoftMax=layers.SoftMax()
    return indiceMax

layer1=layers.ConvLayer(24,3,1,1,3,64,weights_conv_1,biases_conv_1)
layer2=layers.ReluLayer(24,64)
layer3=layers.MaxpoolLayer(24,3,2,1,64)
layer4=layers.ConvLayer(12,3,1,1,64,32,weights_conv_2,biases_conv_2)
layer5=layers.ReluLayer(12,32)
layer6=layers.MaxpoolLayer(12,3,2,1,32)
layer7=layers.ConvLayer(6,3,1,1,32,20,weights_conv_3,biases_conv_3)
layer8=layers.ReluLayer(6,20)
layer9=layers.MaxpoolLayer(6,3,2,1,20)
layer10=layers.FullyConnected(3,10,20,weights_fc,biases_fc)
#f=open("cifar-10-batches-py/data_batch_1")
import pickle
with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
    data_dict = pickle.load(fo, encoding='bytes')
data=data_dict[b'data']
labels=data_dict[b'labels']
image=[[[0 for i in range (3)] for j in range (32)] for k in range (32)]
volume=[[[0 for i in range (24)] for j in range (24)] for k in range (3)]
successCounter=0
nbImages=int(input("Nombre d'images à analyser ?"))
for i in range(min(nbImages,10000)):
    print("Analyse de l'image "+str(i)+"...")
    for j in range(3):
        for k in range(32):
            for l in range(32):
                image[k][l][j]=data[i][j*1024+k*24+l]
    #pic=np.array(image).astype(np.uint8)
    pic=np.array(data[i])
    pic_reshaped=np.transpose(pic.reshape(3,32,32),(1,2,0))
    #print(pic)
    im=Image.fromarray(pic_reshaped)
    im.save("img_origine.png")
    for j in range(24):
        for k in range(24):
            for l in range(3):
                #volume[l][j][k]=image[j+4][k+4][l]
                volume[l][j][k]=pic_reshaped[j+4][k+4][l]
    volume=Normalize(volume)
    result=RunCNN(volume,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8,layer9,layer10)
    print("Résultat calculé :")
    print(result)
    print("Résultat attendu :")
    print(labels[i])
    if(result==labels[i]):
        successCounter+=1
    print("Taux de succès actuel : "+str(successCounter/(i+1))+".")

with open("cifar-10-batches-py/batches.meta",'rb') as fo:
    labels = pickle.load(fo, encoding='bytes')
label_names=labels[b'label_names']
print("Noms des labels :")
print(label_names)
print("Nombre de succès :")
print(successCounter)
print("Taux de succès :")
print(successCounter/(min(nbImages,10000)))

#pic3=np.array(volume3[0]).astype(np.uint8)
#im3=Image.fromarray(pic3)
#im3.save("cat_final_0.png")

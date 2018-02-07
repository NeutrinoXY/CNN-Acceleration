from math import *

class ConvLayer():
    def __init__(self,inputSize,fieldSize,stride,zeroPadding,inputChannels,outputChannels,weights,biases):
        self.inputSize=inputSize
        self.fieldSize=fieldSize
        self.stride=stride
        self.zeroPadding=zeroPadding
        self.inputChannels=inputChannels
        self.outputChannels=outputChannels
        self.weights=weights
        self.biases=biases
        self.nbNeurons=int((self.inputSize-self.fieldSize+2*self.zeroPadding)/self.stride + 1)

    def forward(self,inputVolume):
        tab=[[[0 for i in range(self.nbNeurons)] for j in range(self.nbNeurons)] for k in range(self.outputChannels)]
        inputVolume2=[[[0 for i in range(self.inputSize+2*self.zeroPadding)] for j in range(self.inputSize+2*self.zeroPadding)] for k in range(self.inputChannels)]
        for i in range(self.inputChannels):
            for j in range(self.inputSize):
                for k in range(self.inputSize):
                    inputVolume2[i][j+self.zeroPadding][k+self.zeroPadding]=inputVolume[i][j][k]
        for i in range(self.outputChannels):
            for j in range(self.nbNeurons):
                for k in range(self.nbNeurons):
                    for l in range(self.fieldSize ):
                        for m in range(self.fieldSize):
                            for n in range(self.inputChannels):
                                tab[i][j][k]+=inputVolume2[n][j*self.stride+l][k*self.stride+m]*self.weights[l][m][n][i]
                    tab[i][j][k]+=self.biases[i] # Add a dimension for output channels
        return tab

class MaxpoolLayer():
    def __init__(self,inputSize,fieldSize,stride,zeroPadding,inputChannels):
        self.inputSize=inputSize
        self.fieldSize=fieldSize
        self.stride=stride
        self.zeroPadding=zeroPadding
        self.inputChannels=inputChannels
        self.nbNeurons=int((self.inputSize-self.fieldSize+self.zeroPadding)/self.stride +1)
    def forward(self,inputVolume):
        tab=[[[0 for i in range(self.inputSize)] for j in range(self.inputSize)] for k in range(self.inputChannels)]
        for i in range(len(inputVolume)):
            for j in range(self.zeroPadding):
                for k in range(len(inputVolume[i])):
                    inputVolume[i][k].append(0)
                inputVolume[i].append([0 for k in range(len(inputVolume[i][0]))])
        for i in range(self.inputChannels):
            for j in range(self.nbNeurons):
                for k in range(self.nbNeurons):
                    for l in range(self.fieldSize):
                        for m in range(self.fieldSize):
                            tab[i][j][k]=max(tab[i][j][k],inputVolume[i][j*self.stride+l][k*self.stride+m])
        return tab
        

class ReluLayer():
    def __init__(self,inputSize,inputChannels):
        self.inputSize=inputSize
        self.inputChannels=inputChannels

    def forward(self,inputVolume):
        tab=[[[0 for i in range(self.inputSize)] for j in range(self.inputSize)] for k in range(self.inputChannels)]
        for i in range(self.inputChannels):
            for j in range(self.inputSize):
                for k in range(self.inputSize):
                        tab[i][j][k]=max(inputVolume[i][j][k],0)
        return tab

class FullyConnected():
    def __init__(self,inputSize,outputSize,inputChannels,weights,biases):
        self.inputSize=inputSize
        self.outputSize=outputSize
        self.inputChannels=inputChannels
        self.weights=weights
        self.biases=biases

    def forward(self,inputVolume):
        print(inputVolume)
        tab=[0 for i in range(self.inputSize*self.inputSize*self.inputChannels)]
        for i in range(self.inputChannels):
            for j in range(self.inputSize):
                for k in range(self.inputSize):
                    tab[self.inputSize*(i*self.inputSize+j)+k]=inputVolume[i][j][k]
        #for i in range(self.inputSize):
            #for j in range(self.inputSize):
                #for k in range(self.inputChannels):
                   #tab[self.inputSize*(i*self.inputChannels+j)+k]=inputVolume[k][i][j]
        print(tab)
        tab2=[0 for i in range(self.outputSize)]
        for i in range(self.outputSize):
            for j in range(len(tab)):
                tab2[i]+=tab[j]*self.weights[j][i]
            tab2[i]+=self.biases[i]
        return tab2

class SoftMax():
    def forward(self,inputRow):
        tab=[0 for i in range(len(inputRow))]
        sumSoft=0
        for i in range(len(inputRow)):
            sumSoft+=exp(inputRow[i])
        for i in range(len(inputRow)):
            tab[i]=exp(inputRow[i])/sumSoft
        return tab


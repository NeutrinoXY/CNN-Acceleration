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
        for i in range(self.outputChannels):
            for j in range(self.nbNeurons):
                for k in range(self.nbNeurons):
                    for l in range(self.fieldSize ):
                        for m in range(self.fieldSize):
                            for n in range(self.inputChannels):
                                tab[i][j][k]+=inputVolume[n][j*self.stride+l][k*self.stride+m]*self.weights[l][m][n][i] + self.biases[i] # Add a dimension for output channels
        return tab

class MaxpoolLayer():
    def __init__(self,inputSize,fieldSize,stride,zeropadding,inputChannels):
        self.inputSize=inputSize
        self.fieldSize=fieldSize
        self.stride=stride
        self.inputChannels=inputChannels
        self.nbNeurons=int((self.inputSize-self.fieldSize)/self.stride +1)
    def forward(self,inputVolume):
        tab=[[[0 for i in range(self.inputSize)] for j in range(self.inputSize)] for k in range(self.inputChannels)]
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
    def __init__(self,inputSize,outputSize,inputChannels,weights):
        self.inputSize=inputSize
        self.outputSize
        self.inputChannels=inputChannels
        self.weights=weights

    def forward(self,inputVolume):
        tab=[0 for i in range(self.inputSize*self.inputSize*self.inputChannels)]
        for i in range(self.inputChannels):
            for j in range(self.inputSize):
                for k in range(self.inputSize):
                    tab[self.inputSize(i*self.inputSize+j)+k]=inputVolume[i][j][k]
        tab2=[0 for i in range(outputSize)]
        for i in range(outputSize):
            for j in range(len(tab)):
                tab2[i]+=tab[j]*weights[i][j]
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


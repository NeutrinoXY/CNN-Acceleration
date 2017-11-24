class ConvLayer():
    def __init__(self,inputSize,fieldSize,stride,zeroPadding,inputChannels,outputChannels):
        self.inputSize=inputSize
        self.fieldSize=fieldSize
        self.stride=stride
        self.zeroPadding=zeroPadding
        self.inputChannels=inputChannels
        self.outputChannels=outputChannels

    def forward(inputVolume):
        nbNeurons=(len(inputVolume)-fieldSize+2*zeroPadding)/stride + 1
        tab=[[0 for i in range(nbNeurons)] for j in range(nbNeurons)]
        for i in range(outputChannels):
            for j in range(nbNeurons):
                for k in range(nbNeurons):
                    for l in range(fieldSize ):
                        for m in range(fieldSize):
                            for n in range(inputChannels):
                                tab[i][j][k]+=inputVolume[n][j*stride+l][k*stride+m]*weights[l][m]
        return tab

